from fastapi import HTTPException
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.prompts import PromptTemplate


def get_data_from_url(url: str):
    if "youtube.com" in url:
        loader = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=True)
    else:
        url_list = []
        url_list.append(url)
        loader = UnstructuredURLLoader(
            urls=url_list,
            ssl_verify=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
            },
        )

    data = loader.load()
    if not data:
        raise HTTPException(status_code=404, detail="Data not found")

    return data


def get_prompt_template(summary_type: str, model: str, data: str):

    prompt = f"""
    Generate a summary based on the specified summary type. If the summary type is "abstract", provide a summary in a single paragraph consisting of 250 to 300 words. If the summary type is "list", provide the summary in bullet points.

    Given the text:

    {{text}}

    Summary type: {summary_type}
    """

    if model == "gpt-3.5-turbo-1106":
        prompt_template = PromptTemplate(template=prompt, input_variables=["text"])
    elif model == "bart-large-text-summarizer":
        prompt_template = f"""
            Generate a summary based on the specified summary type. If the summary type is "abstract", provide a summary in a single paragraph consisting of 250 to 300 words. If the summary type is "list", provide the summary in bullet points.

            Given the text:

            {data}

            Summary type: {summary_type}
            """

    return prompt_template
