# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List

import gpt_2_simple as gpt2

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.sess = gpt2.start_tf_sess()
        self.run_name = "gpt2-wow"
        gpt2.load_gpt2(self.sess, run_name=self.run_name)

    def predict(self,
            quest_title: str = Input(description="Title of the quest for which the model should generate dialogue", default="The Lost Kitten"),
            quest_objective: str = Input(description="Objective of the quest for which the model should generate dialogue", default="Search the bustling streets of Dalaran to find a mischievous Fel Kitten that escaped from a warlock's tower."),
            nr_of_samples: int = Input(description="Number of samples to generate", default=1, ge=0),
            batch_size: int = Input(description="Samples per batch. Must divide the total amount of samples.", default=1, ge=1),
            temperature: float = Input(description="Temperature for generation", default=0.9, ge=0, le=1)
    ) -> List[str]:
        """Run a single prediction on the model"""
        if nr_of_samples % batch_size != 0:
            raise ValueError("nr_of_samples must be a multiple of batch_size")
        # pre-processing
        prompt = f"<|startoftext|>{quest_title.strip()}<|obj|>{quest_objective.strip()}<|text|>"
        # generation
        completions = gpt2.generate(self.sess,
              run_name=self.run_name,
              temperature=temperature,
              prefix=prompt,
              truncate="<|endoftext|>",
              nsamples=nr_of_samples,
              batch_size=batch_size,
              include_prefix=False,
              return_as_list=True
              )
        # ... post-processing ...
        return completions

