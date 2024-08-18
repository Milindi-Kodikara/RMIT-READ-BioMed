
import abc
import os
from openai import AzureOpenAI


class Model(abc.ABC):
    @abc.abstractmethod
    def generate_completions(self, input_texts: list[str], prompts: list[str], task: ...) -> list[str]:
        pass


class GPTModel(Model):
    def __init__(self):
        self.deployment_name = os.environ["DEPLOYMENT-NAME"]

        self.client = AzureOpenAI(
            api_key=os.environ["API-KEY"],
            api_version=os.environ["API-VERSION"],
            azure_endpoint=os.environ["ENDPOINT"]
        )

    def generate_completions(self, input_texts: list[str], prompts: list[str], task: ...) -> list[str]:
        self.client.chat.completions.create(model=self.deployment_name, messages=[{"role": "user", "content": "Hello, World!"}])
        pass

    def __call__(self, *args, **kwargs):
        pass


MODELS: dict[str, type[Model]] = {
    "gpt": GPTModel
}


def get_model(model_id: str) -> Model:
    # Only instantiate the model here, because if we do it in MODELS,
    # models without necessary env vars will error
    model_class = MODELS[model_id]
    model_instance = model_class()

    return model_instance
