import asyncio
import json
import httpx


async def main():
    with open("raw/manifest.json", "r") as f:
        files = json.load(f)

    async with httpx.AsyncClient() as client:
        promises = [client.get(filename) for filename in files.keys()]
        responses = await asyncio.gather(*promises)
        for response in responses:
            if response.status_code != 200:
                print(
                    f"Failed to download: {response.url} with status code {response.status_code}"
                )
                continue

            print(f"Downloaded: {response.url}")
            with open(f"raw/{response.url.split('/')[-1]}", "wb") as f:
                f.write(response.content)


if __name__ == "__main__":
    asyncio.run(main())
