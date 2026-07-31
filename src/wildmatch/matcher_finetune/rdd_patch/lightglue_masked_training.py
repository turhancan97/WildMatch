"""
Training-oriented copy of rdd_patch/lightglue_masked.py (itself modified from
https://github.com/cvg/LightGlue).

Kept as a separate module so the inference path in lightglue_masked.py stays
frozen; LightGlueForTraining is a drop-in replacement for LightGlueMasked that
adds one thing: a match score that is always differentiable.

LightGlueMasked reports its matches as `matches`/`scores` — ragged, per-batch-item
lists holding only the entries that cleared `filter_threshold`. A pair with no
surviving match yields an *empty* tensor, and callers that fall back to a
literal zero for it (see _lg_scores in contrastive_finetuning/train_common.py)
end up with a loss term that has no grad_fn. Backward then fires no gradient
hook at all, so under DDP the reducer never finishes the iteration and the next
step aborts with "Expected to have finished reduction in the prior iteration
before starting a new one" — or, with mismatched ranks, hangs in allreduce.

The dense fields sidestep that entirely. `matching_scores0`/`matching_scores1`
are (B, M)/(B, N) and are produced by a `torch.where` whose other branch is
differentiable, so they carry a grad_fn even when nothing matched; `valid0`/
`valid1` are the boolean masks marking which entries cleared the threshold.
Per-pair confidence is then

    (pred["matching_scores0"] * pred["valid0"]).sum(dim=1)

which equals `pred["scores"][i].sum()` exactly — same selection, same values,
only the summation order differs — but stays attached to the graph when the
mask is all-False. Multiplying by an all-zero mask still propagates zeros to
every parameter, which is precisely what keeps DDP's hooks firing.

`matches`/`scores` are still returned unchanged, so eval code that counts
matches per pair keeps working.

A third field, `assignment_scores`, goes one layer further: it's the dense
(B, M+1, N+1) log-assignment matrix `MatchAssignment`/
`sigmoid_log_double_softmax` produce, returned *before* `filter_matches`
touches it at all. `matching_scores0`/`valid0` survive an empty pair only in
aggregate (the all-False mask still has a grad_fn); a single entry that fails
`filter_matches`' mutual-nearest-neighbor check is a hard, gradient-free zero
(`torch.where(mutual0, max0_exp, zero)` with `zero` a fresh constant) no
matter what consumes it downstream. `assignment_scores[b, i, j]` isn't
gated by that check at all — indexing a specific (i, j) is a plain read of an
already-differentiable tensor. That's what
contrastive_finetuning/train_by_lg_matches.py's `distill_correspondence_loss`
needs: a way to push a *specific* currently-unmatched query point toward a
*specific* candidate (from a frozen reference model's own matches, used as a
pseudo-label) even when every point in the pair currently loses the mutual
check — i.e. even when matching_scores0/valid0 alone would give zero
gradient for that pair, full stop.
"""

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.modules.mha import FlashCrossAttention
except ModuleNotFoundError:
    FlashCrossAttention = None

if FlashCrossAttention or hasattr(F, "scaled_dot_product_attention"):
    FLASH_AVAILABLE = True
else:
    FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(
        *x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


def _pad_safe_attn_mask(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A query row whose mask allows no key at all (only padded slots are ever in
    that state) makes the attention softmax run over an empty set: NaN in the
    forward and — the real problem — NaN in scaled_dot_product_attention's
    *backward*, which survives any downstream fixup because 0 * nan = nan
    inside the softmax gradient. The nan_to_num calls this helper pairs with
    only ever repaired the forward, so a batch with any keypoint-count
    imbalance (padding) silently poisoned every gradient; with uniform
    keypoint counts (e.g. top_k always saturated) the dead rows never occurred,
    which is why training worked.

    Returns (safe_mask, dead): `safe_mask` opens dead rows to every key so the
    softmax is well-defined end to end; `dead` marks those rows so the caller
    can zero their output — the exact 0.0 the old nan_to_num produced, now
    with an exactly-zero (instead of NaN) gradient. Real rows are untouched:
    `mask | dead` changes nothing where the row already allows a key.
    """
    dead = ~mask.any(dim=-1, keepdim=True)
    return mask | dead, dead


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        self.has_sdp = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and FlashCrossAttention:
            self.flash_ = FlashCrossAttention()
        if self.has_sdp:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q.shape[-2] == 0 or k.shape[-2] == 0:
            return q.new_zeros((*q.shape[:-1], v.shape[-1]))
        safe_mask = dead = None
        if mask is not None:
            # See _pad_safe_attn_mask: fully-masked (padded) rows would NaN
            # the softmax backward; open them here, zero their output below.
            safe_mask, dead = _pad_safe_attn_mask(mask)
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if self.has_sdp:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=safe_mask).to(q.dtype)
                return v if mask is None else v.masked_fill(dead, 0.0)
            else:
                assert mask is None
                q, k, v = [x.transpose(-2, -3).contiguous() for x in [q, k, v]]
                m = self.flash_(q.half(), torch.stack([k, v], 2).half())
                return m.transpose(-2, -3).to(q.dtype).clone()
        elif self.has_sdp:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=safe_mask)
            return v if mask is None else v.masked_fill(dead, 0.0)
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim = sim.masked_fill(~safe_mask, -float("inf"))
            attn = F.softmax(sim, -1)
            out = torch.einsum("...ij,...jd->...id", attn, v)
            return out if mask is None else out.masked_fill(dead, 0.0)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                # Two separately-masked views of `sim`: opening a dead row for
                # one softmax direction must not leak those entries into the
                # other direction's normalization (see _pad_safe_attn_mask).
                safe01, dead0 = _pad_safe_attn_mask(mask)
                safe10, dead1 = _pad_safe_attn_mask(mask.transpose(-1, -2))
                attn01 = F.softmax(sim.masked_fill(~safe01, -float("inf")), dim=-1)
                attn10 = F.softmax(
                    sim.transpose(-2, -1).contiguous().masked_fill(~safe10, -float("inf")),
                    dim=-1,
                )
            else:
                attn01 = F.softmax(sim, dim=-1)
                attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.masked_fill(dead0, 0.0), m1.masked_fill(dead1, 0.0)
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    # valid0/valid1 are handed back rather than left to be recovered as
    # `m0 > -1`: identical content, but the dense path then never has to undo
    # the -1 sentinel encoding just to get a mask that was already computed.
    return m0, m1, mscores0, mscores1, valid0, valid1


class LightGlueForTraining(nn.Module):
    """LightGlueMasked plus always-differentiable dense match scores.

    Identical architecture, weights and numerics; see the module docstring for
    why the dense `matching_scores*` / `valid*` outputs exist and how to turn
    them into a per-pair confidence.
    """

    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.01,  # match threshold
        "weights": None,
        "detach_descriptors": True,  # cut gradient into the descriptor extractor (e.g. RDD)
    }

    # Point pruning involves an overhead (gather).
    # Therefore, we only activate it if there are enough keypoints.
    pruning_keypoint_thresholds = {
        "cpu": -1,
        "mps": -1,
        "cuda": 1024,
        "flash": 1536,
    }

    required_data_keys = ["image0", "image1"]

    version = "v0.1_arxiv"
    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    features = {
        "superpoint": {
            "weights": "superpoint_lightglue",
            "input_dim": 256,
        },
        "disk": {
            "weights": "disk_lightglue",
            "input_dim": 128,
        },
        "aliked": {
            "weights": "aliked_lightglue",
            "input_dim": 128,
        },
        "sift": {
            "weights": "sift_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
        "doghardnet": {
            "weights": "doghardnet_lightglue",
            "input_dim": 128,
            "add_scale_ori": True,
        },
        "rdd": {
            "weights": './weights/RDD_lg-v2.pth',
            "input_dim": 256,
        },
    }

    def __init__(self, features="rdd", **conf) -> None:
        super().__init__()
        feature_conf = {}
        if features is not None:
            if features not in self.features:
                raise ValueError(
                    f"Unsupported features: {features} not in "
                    f"{{{','.join(self.features)}}}"
                )
            feature_conf = self.features[features]

        merged_conf = {
            **self.default_conf,
            **feature_conf,
            **conf,
        }

        self.conf = conf = SimpleNamespace(**merged_conf)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * self.conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor(
                [self.confidence_threshold(i) for i in range(self.conf.n_layers)]
            ),
        )

        state_dict = None
        if features is not None and features != 'rdd':
            fname = f"{conf.weights}_{self.version.replace('.', '-')}.pth"
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format(self.version, features), file_name=fname
            )
            self.load_state_dict(state_dict, strict=False)
        elif conf.weights is not None:
            if features == 'rdd':
                path = Path(conf.weights)
            else:
                path = Path(__file__).parent
                path = path / "weights/{}.pth".format(self.conf.weights)
            state_dict = torch.load(str(path), map_location="cpu")

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

        # static lengths LightGlue is compiled for (only used with torch.compile)
        self.static_lengths = None

    def compile(
        self, mode="reduce-overhead", static_lengths=[256, 512, 768, 1024, 1280, 1536]
    ):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        torch._inductor.cudagraph_mark_step_begin()
        for i in range(self.conf.n_layers):
            self.transformers[i].masked_forward = torch.compile(
                self.transformers[i].masked_forward, mode=mode, fullgraph=True
            )

        self.static_lengths = static_lengths

    def forward(self, data: dict) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            image0: dict
                keypoints: [B x M x 2]
                descriptors: [B x M x D]
                image: [B x C x H x W] or image_size: [B x 2]
                masks: [B x M x 1]
            image1: dict
                keypoints: [B x N x 2]
                descriptors: [B x N x D]
                image: [B x C x H x W] or image_size: [B x 2]
                masks: [B x N x 1]
        Output (dict):
            matches0: [B x M]          index of the matched keypoint, -1 if none
            matching_scores0: [B x M]  dense confidence, always differentiable
            valid0: [B x M]            bool, which entries cleared the threshold
            matches1: [B x N]
            matching_scores1: [B x N]
            valid1: [B x N]
            matches: List[[Si x 2]]    ragged, threshold-filtered (legacy)
            scores: List[[Si]]         ragged, threshold-filtered (legacy)
            assignment_scores: [B x (M+1) x (N+1)]  dense log-assignment,
                                pre-filter_matches — see module docstring
            stop: int
            prune0: [B x M]
            prune1: [B x N]

        For a loss, use the dense pair:
        `(matching_scores0 * valid0).sum(dim=1)` — same value as
        `scores[i].sum()`, but it survives a pair with zero matches with its
        graph intact. See the module docstring.
        """
        with torch.autocast(enabled=self.conf.mp, device_type="cuda"):
            return self._forward(data)

    def _forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data0["keypoints"], data1["keypoints"]
        
        mask0, mask1 = data0["masks"], data1["masks"]

        if kpts0.dim() == mask0.dim():
            mask0 = mask0.unsqueeze(1)
        if kpts1.dim() == mask1.dim():
            mask1 = mask1.unsqueeze(1)

        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            kpts0 = torch.cat(
                [kpts0] + [data0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
            kpts1 = torch.cat(
                [kpts1] + [data1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
        if self.conf.detach_descriptors:
            desc0 = data0["descriptors"].detach().contiguous()
            desc1 = data1["descriptors"].detach().contiguous()
        else:
            desc0 = data0["descriptors"].contiguous()
            desc1 = data1["descriptors"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim

        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()

        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = False
        do_point_pruning = False
        pruning_th = self.pruning_min_kpts(device)
        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
                break
            desc0, desc1 = self.transformers[i](
                desc0, desc1, encoding0, encoding1, mask0=mask0, mask1=mask1
            )
            if i == self.conf.n_layers - 1:
                continue  # no early stopping or adaptive width at last layer

            if do_early_stop:
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m], token1[..., :n], i, m + n):
                    break
            if do_point_pruning and desc0.shape[-2] > pruning_th:
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
            if do_point_pruning and desc1.shape[-2] > pruning_th:
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        if desc0.shape[1] == 0 or desc1.shape[1] == 0:  # no keypoints
            m0 = desc0.new_full((b, m), -1, dtype=torch.long)
            m1 = desc1.new_full((b, n), -1, dtype=torch.long)
            # The one place the dense outputs cannot inherit a graph on their
            # own: there is no assignment matrix to differentiate, so
            # LightGlueMasked's bare new_zeros are genuinely detached. `anchor`
            # is identically zero — no value changes anywhere — but it restores
            # a path from these outputs back to LightGlue's own weights and to
            # whatever produced the descriptors, so backward still reaches the
            # parameters and DDP's reducer still completes. Summing an empty
            # tensor keeps its grad_fn, which is what makes the desc0 term work
            # even though desc0 has zero keypoints. Anchoring only the last
            # MatchAssignment (the sole log_assignment layer this forward ever
            # applies) is enough because DDP runs with
            # find_unused_parameters=True and marks the rest ready without
            # waiting.
            anchor = desc0.sum() * 0.0
            for p in self.log_assignment[-1].parameters():
                anchor = anchor + p.sum() * 0.0
            mscores0 = desc0.new_zeros((b, m)) + anchor.to(desc0.dtype)
            mscores1 = desc1.new_zeros((b, n)) + anchor.to(desc1.dtype)
            valid0 = torch.zeros((b, m), dtype=torch.bool, device=device)
            valid1 = torch.zeros((b, n), dtype=torch.bool, device=device)
            matches = desc0.new_empty((b, 0, 2), dtype=torch.long)
            mscores = desc0.new_empty((b, 0))
            # Same anchor trick as mscores0/1 above — no assignment matrix
            # was computed on this branch, so this is a same-shape zero
            # rather than a real log-assignment; distill_correspondence_loss
            # never indexes into it for real, since ref_valid0 for this same
            # (RDD-frozen, hence identical-input) pair is empty too whenever
            # this branch fires — see its docstring.
            assignment_scores = desc0.new_zeros((b, m + 1, n + 1)) + anchor.to(desc0.dtype)
            if not do_point_pruning:
                prune0 = torch.ones_like(mscores0) * self.conf.n_layers
                prune1 = torch.ones_like(mscores1) * self.conf.n_layers
            return {
                "matches0": m0,
                "matches1": m1,
                "matching_scores0": mscores0,
                "matching_scores1": mscores1,
                "valid0": valid0,
                "valid1": valid1,
                "stop": i + 1,
                "matches": matches,
                "scores": mscores,
                "assignment_scores": assignment_scores,
                "prune0": prune0,
                "prune1": prune1,
            }

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]  # remove padding
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1, valid0, valid1 = filter_matches(
            scores, self.conf.filter_threshold
        )
        matches, mscores = [], []
        for k in range(b):
            valid = valid0[k]  # == m0[k] > -1, straight from filter_matches
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])

        # TODO: Remove when hloc switches to the compact format.
        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            # Scattered alongside the scores they index, so `valid*` stays
            # aligned with `matching_scores*` on this branch too. Dead code as
            # long as do_point_pruning is False, but leaving the masks
            # un-remapped here would be a silent trap if it were ever enabled.
            valid0_ = torch.zeros((b, m), dtype=torch.bool, device=valid0.device)
            valid1_ = torch.zeros((b, n), dtype=torch.bool, device=valid1.device)
            valid0_[:, ind0] = valid0
            valid1_[:, ind1] = valid1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
            valid0, valid1 = valid0_, valid1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "valid0": valid0,
            "valid1": valid1,
            "stop": i + 1,
            "matches": matches,
            "scores": mscores,
            "assignment_scores": scores,
            "prune0": prune0,
            "prune1": prune1,
        }

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]
