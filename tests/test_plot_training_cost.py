import unittest

from paper.figures.plot_training_cost import (
    checkpoint_epoch,
    clock_seconds,
    cumulative_gpu_hours,
    job_cumulative_gpu_hours,
    parse_loma_eval_seconds,
    parse_loma_train_seconds,
    parse_probe_epochs,
    parse_tqdm_train_seconds,
)


class ProbeLogTests(unittest.TestCase):
    def test_epoch_lines_give_query_split_metrics(self):
        text = (
            "[linear_probe] epoch 1/50 train_loss=4.4 train_top1=0.15 train_top5=0.33 train_top10=0.43 "
            "val_loss=5.0 val_top1=0.1200 val_top5=0.2500 val_top10=0.3100\n"
            "[linear_probe] epoch 2/50 train_loss=3.5 train_top1=0.27 train_top5=0.49 train_top10=0.60 "
            "val_loss=4.0 val_top1=0.2000 val_top5=0.3000 val_top10=0.4000\n"
        )
        epochs = parse_probe_epochs(text)
        self.assertEqual(sorted(epochs), [1, 2])
        self.assertEqual(epochs[2], {"top_1": 0.2, "top_5": 0.3, "top_10": 0.4})

    def test_balanced_top1_is_parsed_when_the_log_has_it(self):
        text = (
            "[linear_probe] epoch 1/50 train_loss=4.4 train_top1=0.15 train_top5=0.33 train_top10=0.43 "
            "val_loss=5.0 val_top1=0.1200 val_top5=0.2500 val_top10=0.3100 val_balanced_top1=0.0800\n"
        )
        self.assertEqual(parse_probe_epochs(text)[1]["balanced_top_1"], 0.08)

    def test_tqdm_takes_the_final_bar_of_each_epoch(self):
        text = (
            "[linear_probe][train] epoch 1/50:  99%|#### | 1738/1740 [11:58<00:01,  2.42it/s]\r"
            "[linear_probe][train] epoch 1/50: 100%|#####| 1740/1740 [12:03<00:00,  2.41it/s, loss=3.9]\r"
            "[linear_probe][val] epoch 1/50: 100%|#####| 100/100 [01:10<00:00]\n"
            "[linear_probe][train] epoch 2/50: 100%|#####| 1740/1740 [1:02:05<00:00,  2.41it/s]\n"
        )
        self.assertEqual(parse_tqdm_train_seconds(text), {1: 723.0, 2: 3725.0})
        self.assertEqual(clock_seconds("1:00:01"), 3601.0)


class LomaLogTests(unittest.TestCase):
    def test_json_epoch_records_give_training_seconds(self):
        text = (
            '{\n  "epoch": 0,\n  "train/lr": 1e-05,\n  "time/epoch_eval_s": 0.0,\n  "time/train_s": 17.5\n}\n'
            '{\n  "epoch": 1,\n  "time/train_s": 15.0\n}\n'
        )
        self.assertEqual(parse_loma_train_seconds(text), {0: 17.5, 1: 15.0})

    def test_cumulative_gpu_hours_counts_every_gpu_and_rejects_gaps(self):
        hours = cumulative_gpu_hours({0: 900.0, 1: 900.0}, gpus=4)
        self.assertAlmostEqual(hours[0], 1.0)
        self.assertAlmostEqual(hours[1], 2.0)
        with self.assertRaises(ValueError):
            cumulative_gpu_hours({0: 1.0, 2: 1.0}, gpus=1)

    def test_job_cost_charges_overhead_up_front_and_adds_evaluation(self):
        text = '{\n  "epoch": 0,\n  "time/epoch_eval_s": 60.0,\n  "time/train_s": 30.0\n}\n'
        self.assertEqual(parse_loma_eval_seconds(text), {0: 60.0})
        # epoch 0: 900 s training + 900 s evaluation; epoch 1: 900 s training;
        # the remaining 2700 s of the 5400 s job is overhead, charged at epoch 0; 2 GPUs
        hours = job_cumulative_gpu_hours({0: 900.0, 1: 900.0}, {0: 900.0}, 5400.0, gpus=2)
        self.assertAlmostEqual(hours[0], (2700 + 1800) * 2 / 3600)
        self.assertAlmostEqual(hours[1], 5400 * 2 / 3600)
        with self.assertRaises(ValueError):
            job_cumulative_gpu_hours({0: 900.0}, {}, 100.0, gpus=1)

    def test_checkpoint_epoch_is_read_from_the_directory(self):
        self.assertEqual(checkpoint_epoch("/x/loma/epoch_050/model.safetensors"), 50)
        with self.assertRaises(ValueError):
            checkpoint_epoch("/x/loma/latest/model.safetensors")


if __name__ == "__main__":
    unittest.main()
