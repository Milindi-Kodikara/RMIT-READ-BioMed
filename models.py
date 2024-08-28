import abc
import logging
import os
from typing import LiteralString, Any, List, Dict

from openai import AzureOpenAI


class Model(abc.ABC):
    @abc.abstractmethod
    def generate_result(self, prompt: str) -> str:
        pass

    def get_results(self, embedded_prompts) -> list[dict[str, Any]]:
        results = []

        for embedded_prompt in embedded_prompts:
            pmid = embedded_prompt['pmid']

            for prompt_item in embedded_prompt['prompts']:
                prompt_id = prompt_item['prompt_id']
                prompt = prompt_item['prompt']

                print(f'pmid: {pmid}\tprompt_id: {prompt_id}')

                generated_text_result = self.generate_result(prompt)

                results.append({'pmid': pmid, 'prompt_id': prompt_id, 'result': generated_text_result})

        return results


class GPTModel(Model):
    def __init__(self):
        self.deployment_name = os.environ["DEPLOYMENT-NAME"]

        self.client = AzureOpenAI(
            api_key=os.environ["API-KEY"],
            api_version=os.environ["API-VERSION"],
            azure_endpoint=os.environ["ENDPOINT"]
        )
        logging.info("Model initialised.")

    def __call__(self, *args, **kwargs):
        pass

    def generate_result(self, prompt: str) -> str:
        response = self.client.chat.completions.create(model=self.deployment_name,
                                                       messages=[{"role": "user", "content": prompt}])

        generated_text_result = response.choices[0].message.content

        return generated_text_result


MODELS: dict[str, type[Model]] = {
    "GPT": GPTModel
}


def get_model(model_id: str) -> Model:
    # Only instantiate the model here, because if we do it in MODELS,
    # models without necessary env vars will error
    model_class = MODELS[model_id]
    model_instance = model_class()

    return model_instance
