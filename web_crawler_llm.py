from bs4 import BeautifulSoup
import asyncio
from pyppeteer import launch
from pyppeteer_stealth import stealth
from urllib.parse import urlparse, urlunparse
import os
import traceback
import random
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from langchain import HuggingFacePipeline
from huggingface_hub import login
import torch

import api_tokens


urls = ["https://www.quora.com/", 
        "https://www.amazon.co.uk/",
        "https://www.olx.pt/",
        "https://www.target.com/"
        ]

user_agents = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:103.0) Gecko/20100101 Firefox/103.0",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36 Edg/104.0.0.0",
                   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
                   "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
                   "Mozilla/5.0 (iPhone; CPU iPhone OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
                   "Mozilla/5.0 (Linux; Android 11; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Mobile Safari/537.36",
                   "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Mobile Safari/537.36",
                   "Mozilla/5.0 (iPhone; CPU iPhone OS 15_6 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
                   "Mozilla/5.0 (Android 12; Mobile; rv:103.0) Gecko/103.0 Firefox/103.0"]

def make_decision_zero_shot_learning_cot(tokenizer, 
                                         model, 
                                         soup,
                                         pipe):
    sample_prompt = """
    Let's think step by step about the best way to navigate this website in order to create and login to your account.
    """

    input_text = f"{sample_prompt} \nBased on the following webpage content {soup}: \nTell me the next action to perform on the website:"

    sequences = pipe(
        input_text,
        do_sample=True,
        max_new_tokens=100, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95,
        num_return_sequences=1,
    )
    print(f"next_action: {sequences[0]['generated_text']}")

    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    # outputs = model.generate(input_ids)
    # next_action = tokenizer.decode(outputs[0])
    # print(f"next_action: {next_action}")

    # return next_action
    return sequences[0]['generated_text']

async def get_page_content(browser, url, user_agent) -> str:
    page = await browser.newPage()

    # Look at documentation: https://pypi.org/project/pyppeteer-stealth/
    await stealth(page, user_agent=user_agent)

    # Urls need to be prefixed with http:// or https://
    await page.goto(url,
                    wait_until='networkidle0', # Wait until there are no more than 0 network connections for at least 500 ms
                    timeout= 30000) # Throws timeout exception after 30 seconds

    await page.waitFor(3000) # Wait for 2 seconds for page to load
    html = await page.content()
    soup = BeautifulSoup(html, 'html.parser')

    return page, soup

async def visit_page(url: str, 
                     user_agent: str,
                     tokenizer, 
                     model,
                     pipe):
    # Add http:// if not in url
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        #parsed_url = parsed_url._replace(scheme="http")
        parsed_url = parsed_url._replace(scheme="https")
    url = urlunparse(parsed_url)
    if '///' in url:
        url = url.replace('///', '//')

    browser = await launch(
                        ignoreHTTPSErrors=True, # This prevents browser from hanging due to HTTPS blocking insecure content
                        devtools=True, 
                        dumpio=True, # Essential to log when browser hangs, this will print the errors
                        args=['--no-sandbox', 
                              '--disable-setuid-sandbox', 
                              '--disable-gpu',
                              '--headless',
                              '--mute-audio'])

    page, soup = get_page_content(browser, url, user_agent)

    next_action = make_decision_zero_shot_learning_cot(tokenizer, model, soup, pipe)

    for i in range(0, 10):
        try:
            await page.waitForSelector(next_action, timeout=5000)
            await page.click(next_action)
            break
        except:
            print(f"Failed to click on {next_action}, retrying...")
            await page.waitFor(3000)
            continue

    await browser.close()

def crawl_website(url: str,
               user_agent: str,
               tokenizer, 
               model,
               pipe):
    try:
        print(f"\n\n------- Website {url} --------")
        asyncio.run(visit_page(url, user_agent, tokenizer, model, pipe))
        # In Jupyter notebooks, use the following instead of asyncio.run
        # this happens because Jupyter already has an event loop running: 
        # https://stackoverflow.com/questions/47518874/how-do-i-run-python-asyncio-code-in-a-jupyter-notebook
        # loop = asyncio.get_event_loop()
        # loop.create_task(visit_page(url, user_agent, tokenizer, model))

    except Exception as e:
        print(f"Error running command: {e}")
        traceback.print_exc()

def build_langchain_pipeline(model_name: str) -> (AutoTokenizer,
                                                  AutoModelForCausalLM,
                                                  HuggingFacePipeline):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
                                                
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map='auto',
                                                torch_dtype=torch.float16,
                                                quantization_config=bnb_config,
                                                load_in_4bit=True,
                                                trust_remote_code=True,
                                                )

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer= tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    # max_new_tokens = 1024,
                    # do_sample=True,
                    # top_k=10,
                    # num_return_sequences=1,
                    # eos_token_id=tokenizer.eos_token_id
                    )

    llm = HuggingFacePipeline(pipeline=pipe,
                                model_kwargs={
                                    'temperature':0
                                },
                            )
    
    return tokenizer, model, pipe, llm


def main() -> None:
    os.environ['HF_DATASETS_CACHE'] = api_tokens.HF_DATASETS_CACHE

    #model_name = 'meta-llama/Llama-2-7b-chat-hf'
    #model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    model_name = "Trelis/Llama-2-7b-chat-hf-sharded-bf16"

    login(token=api_tokens.HUGGINGFACEHUB_API_TOKEN)
    tokenizer, model, pipe, llm = build_langchain_pipeline(model_name)

    for url in urls:
        user_agent = random.choice(user_agents)
        crawl_website(url, user_agent, tokenizer, model, pipe)


if __name__ == "__main__":
    main()