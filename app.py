"""
Code to be run in each docker container to collect a dataset of websites
"""

import requests
import sys
import time
import asyncio
from pyppeteer import launch
from pyppeteer_stealth import stealth
from urllib.parse import urlparse, urlunparse
import os


async def intercept_response(response, output_directory: str) -> None:
    """ Inspired on https://maxiee.github.io/post/PyppeteerCrawlingInterceptResponsemd/
        Mirrors all resources associated to a page by intercepting 
        all the responses from the web server.

    Args:
        response: Individual responses received by the web server
        output_directory (str): Directory to save the mirrored website
    """

    print(f"\n\n\n-- RESPONSE url: {response.url}")

    content = await response.buffer()
    if isinstance(content, str):
        content = content.encode('utf-8')

    url = response.url
    parsed_url = urlparse(url)
    file_name = parsed_url.path

    path_components = file_name.split('/')     
    # Remove the first empty element if present
    if path_components[0] == '':
        path_components.pop(0)
    # Construct the subdirectories
    subdirectories = path_components[:-1]
    subdir_path = output_directory
    # Create the subdirectories
    for subdirectory in subdirectories:
        subdir_path = os.path.join(subdir_path, subdirectory)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    if output_directory.endswith('/'):
        output_directory = output_directory[:-1]
    file_path = os.path.join(output_directory, file_name)

    with open(file_path, 'wb') as file:
        file.write(content)


# This functionality seems to be currently disabled in pyppeteer
async def intercept_request(request):
    if request.isNavigationRequest() and len(request.redirectChain()) != 0:
        await request.abort()
    else:
        await request.continue_()


# Some visits fail (non-deterministically), but more commonly
# in certain websites, such as https://www.milanuncios.com/ , 
# https://open.spotify.com/? , and https://www.xfinity.com/national/ ,
# https://www.twitch.tv/ , 
# This seems to be caused by websites containing insecure content, which is blocked in HTTPS
async def visit_page(url: str, 
                     user_agent: str, 
                     screenshot_path: str,
                     website_path: str, 
                     mirror_website: bool=False, 
                     store_screenshot: bool=False, 
                     store_html: bool=False) -> str:
    # Add http:// if not in url
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        #parsed_url = parsed_url._replace(scheme="http")
        parsed_url = parsed_url._replace(scheme="https")
    url = urlunparse(parsed_url)
    if '///' in url:
        url = url.replace('///', '//')

    print("parsed url")

    browser_path = "/usr/bin/chromium-browser"
    #browser = await launch(executablePath=browser_path,
    browser = await launch(
                           ignoreHTTPSErrors=True, # This prevents browser from hanging due to HTTPS blocking insecure content
                           headless=True, 
                           devtools=True, 
                           dumpio=True, # Essential to log when browser hangs, this will print the errors
                           args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu'])

    page = await browser.newPage()

    if mirror_website == True:
        # Set hook to intecept responses from the web server
        page.on('response', 
            lambda response: asyncio.ensure_future(intercept_response(response, website_path)))
    print("stealth")
    # Look at documentation: https://pypi.org/project/pyppeteer-stealth/
    await stealth(page, user_agent=user_agent)

    # Abort in case of chain of redirects, currently disabled in pyppeteer
    #page.on('request',
    #        lambda request: asyncio.ensure_future(intercept_request(request)))
    print("goto")
    # Urls need to be prefixed with http:// or https://
    await page.goto(url,
                    wait_until='networkidle0', # Wait until there are no more than 0 network connections for at least 500 ms
                    timeout= 30000) # Throws timeout exception after 30 seconds

    await page.waitFor(3000) # Wait for 2 seconds for page to load
    #await mirror_page(page, website_path)
    html = await page.content()
    if store_screenshot == True:
        await page.screenshot({'path': f'{screenshot_path}', 'fullPage': True})
    await browser.close()

    if store_html == True:
        with open(os.path.join(website_path, 'index.html'), 'w') as file:
            file.write(html)

    return html

async def main(url: str, 
               user_agent: str, 
               screenshot_path: str, 
               website_path: str, 
               mirror_website: bool, 
               store_screenshot: bool, 
               store_html: bool):
    try:
        await asyncio.wait_for(visit_page(url, 
                                          user_agent, 
                                          screenshot_path, 
                                          website_path, 
                                          mirror_website, 
                                          store_screenshot, 
                                          store_html),
                             timeout=30)  # 30 seconds timeout
    except TimeoutError:
        print('timeout!')


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("ERROR starting app.py!")
        print("Usage: python app.py <url> <user_agent> <screenshot_path> <website_path> <mirror_website> <store_screenshot> <store_html>")
        sys.exit(1)

    url = sys.argv[1]
    user_agent = sys.argv[2]
    screenshot_path = sys.argv[3]
    website_path = sys.argv[4]
    # Transform into boolean
    mirror_website = True if sys.argv[5] == "True" else False
    store_screenshot = True if sys.argv[6] == "True" else False
    store_html = True if sys.argv[7] == "True" else False

    # TODO: FIND WHY THIS BLOCKS IF CALLED DIRECTLY INSTEAD OF RUNNING ON A CONTAINER?????
    asyncio.run(main(url, 
                     user_agent, 
                     screenshot_path, 
                     website_path,
                     mirror_website, 
                     store_screenshot, 
                     store_html)
    )