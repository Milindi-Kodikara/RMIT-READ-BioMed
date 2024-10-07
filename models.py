import abc
import logging
import os
from typing import LiteralString, Any, List, Dict

from openai import AzureOpenAI

import boto3
from botocore.exceptions import ClientError

import json
from dotenv import load_dotenv

load_dotenv()


class Model(abc.ABC):
    @abc.abstractmethod
    def generate_result(self, prompt: str) -> str:
        pass

    def get_results(self, embedded_prompts) -> list[dict[str, Any]]:
        results = []
        all_prompts = len(embedded_prompts)
        count = 1
        for embedded_prompt in embedded_prompts:
            pmid = embedded_prompt['pmid']

            for prompt_item in embedded_prompt['prompts']:
                prompt_id = prompt_item['prompt_id']
                prompt = prompt_item['prompt']

                print(f'Text file number {count} out of {all_prompts} text files\tpmid: {pmid}\tprompt_id: {prompt_id}')
                generated_text_result = self.generate_result(prompt)

                results.append({'pmid': pmid, 'prompt_id': prompt_id, 'result': generated_text_result})
            count = count + 1
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


class LlamaModel(Model):
    def __init__(self):
        profile_name = os.environ["PROFILE-NAME"]
        boto3.setup_default_session(profile_name=profile_name)
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Set the model ID
        self.model_id = os.environ["MODEL-ID"]

    def __call__(self, *args, **kwargs):
        pass

    def generate_result(self, prompt: str) -> str:
        # Embed the prompt in Llama 3's instruction format.
        formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
        }

        # Convert the native request to JSON.
        request = json.dumps(native_request)

        try:
            # Invoke the model with the request.
            response = self.client.invoke_model(modelId=self.model_id, body=request)

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            exit(1)

        # Decode the response body.
        model_response = json.loads(response["body"].read())

        # Extract and return the response text.
        generated_text_result = model_response["generation"]
        return generated_text_result


MODELS: dict[str, type[Model]] = {
    "gpt-35-turbo-16k": GPTModel,
    "meta.llama3-70b-instruct-v1:0": LlamaModel,
}


def get_model(model_id: str) -> Model:
    # Only instantiate the model here, because if we do it in MODELS,
    # models without necessary env vars will error
    model_class = MODELS[model_id]
    model_instance = model_class()

    return model_instance
