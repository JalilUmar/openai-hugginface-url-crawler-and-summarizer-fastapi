from fastapi import FastAPI, status, HTTPException
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv, find_dotenv
from controller import get_data_from_url, get_prompt_template
from transformers import AutoTokenizer
import os
import random
import requests

_: bool = load_dotenv(find_dotenv())

openai_api_key = os.getenv("OPENAI_API_KEY")
ENDPOINT_URL = (
    "https://api-inference.huggingface.co/models/Azma-AI/bart-large-text-summarizer"
)
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hello world !"}


models = [
    {"name": "gpt-3.5-turbo-1106", "api_key": openai_api_key},
    {"name": "bart-large-text-summarizer", "api_key": HF_TOKEN},
]


@app.get("/summarize", status_code=status.HTTP_200_OK)
async def summarize_url_content(url: str, summary_type: str):
    try:
        selected_model = random.choice(models)

        print("Selected model:\t", selected_model["name"])

        data = get_data_from_url(url)

        prompt = get_prompt_template(
            summary_type, model=selected_model["name"], data=data
        )

        if selected_model["name"] == "gpt-3.5-turbo-1106":

            client = ChatOpenAI(
                api_key=selected_model["api_key"], model=selected_model["name"]
            )

            chain = load_summarize_chain(llm=client, prompt=prompt, chain_type="stuff")
            res = chain.invoke(data)
            summary = res["output_text"] or res
        elif selected_model["name"] == "bart-large-text-summarizer":

            tokenizer = AutoTokenizer.from_pretrained(
                "Azma-AI/bart-large-text-summarizer"
            )
            prompt = [{"role": "user", "content": f"{prompt}"}]
            message = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
            )
            req = requests.post(
                ENDPOINT_URL,
                headers={
                    "Authorization": "Bearer " + HF_TOKEN,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": f"""
                            {message}
                        """,
                    "parameters": {
                        "do_sample": False,
                        "truncation": "only_first",
                    },
                },
            )

            res = req.json()
            summary = res[0]["generated_text"] or res

        return {
            "input": {"url": url, "summary_type": summary_type},
            "summary": summary,
            "llm": selected_model["name"],
        }

    except HTTPException as e:
        raise e
