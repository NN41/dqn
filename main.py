from src.config import Config
from src.trainer import Trainer

config = Config(
    num_samples_per_epoch=10_000,
    num_epochs=1,
    update_interval=1000
)
trainer = Trainer(config)
trainer.train()