from src.config import Config
from src.trainer import Trainer

config = Config(num_test_frames=5, num_samples_per_epoch=100, num_epochs=10)
trainer = Trainer(config)
trainer.train()
