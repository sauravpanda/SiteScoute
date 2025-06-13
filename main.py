import asyncio
import os
import sys
import time
from datetime import datetime
import json
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from browser_use import Agent, BrowserProfile
from browser_use.browser.session import BrowserSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sitescout.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Dictionary of websites to check
WEBSITES = {
    "Entertainment & Media": {
        "Spotify": "https://open.spotify.com",
        "Snapchat": "https://www.snapchat.com",
        "Discord": "https://discord.com",
        "FuboTV": "https://www.fubo.tv",
        "Vimeo": "https://vimeo.com"
    },
    "Gaming": {
        "Pokémon Go": "https://pokemongolive.com",
        "Pokemon TCG": "https://www.pokemon.com/us/pokemon-tcg",
        "Phasmophobia": "https://store.steampowered.com/app/739630/Phasmophobia",
        "Rocket League": "https://www.rocketleague.com",
        "Roblox": "https://www.roblox.com",
        "Dragon Ball": "https://www.toei-animation.com/dragonball",
        "Marvel Contest of Champions": "https://playcontestofchampions.com"
    },
    "AI & Technology Platforms": {
        "CharacterAI": "https://character.ai",
        "Anthropic": "https://www.anthropic.com",
        "OpenAI": "https://openai.com",
        "Cursor": "https://www.cursor.com",
        "Google Gemini": "https://gemini.google.com"
    },
    "Google Services": {
        "Google": "https://www.google.com",
        "Google Cloud": "https://cloud.google.com",
        "Google Meet": "https://meet.google.com",
        "Gmail": "https://gmail.com",
        "Google Nest": "https://store.google.com/category/connected_home",
        "Google Maps": "https://maps.google.com"
    },
    "Cloud & Infrastructure": {
        "Amazon Web Services": "https://aws.amazon.com",
        "Microsoft Azure": "https://azure.microsoft.com",
        "Microsoft 365": "https://www.microsoft365.com",
        "Cloudflare": "https://www.cloudflare.com",
        "Box": "https://www.box.com",
        "NPM": "https://www.npmjs.com"
    },
    "E-commerce & Business": {
        "Etsy": "https://www.etsy.com",
        "Shopify": "https://www.shopify.com",
        "DoorDash": "https://www.doordash.com",
        "Wayfair": "https://www.wayfair.com"
    },
    "Business Tools & Services": {
        "UPS": "https://www.ups.com",
        "USPS": "https://www.usps.com",
        "T-Mobile": "https://www.t-mobile.com",
        "Mailchimp": "https://mailchimp.com",
        "Dialpad": "https://www.dialpad.com",
        "Zoom": "https://zoom.us",
        "Calendly": "https://calendly.com"
    },
    "Smart Home & IoT": {
        "Ecobee": "https://www.ecobee.com",
        "Fitbit": "https://www.fitbit.com"
    },
    "Specialized Business Software": {
        "HighLevel": "https://www.gohighlevel.com",
        "Clover POS Systems": "https://www.clover.com",
        "Procore": "https://www.procore.com"
    },
    "Education & Development": {
        "Khan Academy": "https://www.khanacademy.org",
        "DeviantArt": "https://www.deviantart.com"
    },
    "Finance & Banking": {
        "Dave": "https://dave.com"
    }
}

async def check_website(llm: ChatOllama, url: str, name: str, browser_profile: BrowserProfile) -> tuple[str, dict]:
    logger.info(f"Checking website: {name} ({url})")
    task = f"""Visit {url} and check if the website is working properly.\nSimply respond with a JSON:\n{{\n    \"status\": \"UP | DOWN\",\n    \"reason\": \"brief explanation of what you see\"\n}}\n"""
    browser_session = BrowserSession(browser_profile=browser_profile, headless=False)
    try:
        await browser_session.start()
        page = await browser_session.new_tab()
        try:
            agent = Agent(
                task=task,
                browser_session=browser_session,
                page=page,
                llm=llm
            )
            result = await agent.run()
            last_message = result.final_result() if result else ""
            logger.debug(f"Agent response for {name}: {last_message[:200]}...")
            try:
                analysis = json.loads(last_message)
                is_up = analysis.get('status') == 'UP'
                status = "UP" if is_up else "DOWN"
                logger.info(f"Website {name} status: {status}")
                return name, {
                    "status": status,
                    "url": url,
                    "error": analysis.get('reason', 'No reason provided') if not is_up else None
                }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response for {name}: {str(e)}")
                return name, {
                    "status": "DOWN",
                    "url": url,
                    "error": f"Invalid response format: {last_message}"
                }
        finally:
            await page.close()
    except Exception as e:
        logger.error(f"Error checking {name}: {str(e)}", exc_info=True)
        return name, {
            "status": "DOWN",
            "url": url,
            "error": str(e)
        }
    finally:
        await browser_session.close()

def process_batch(llm: ChatOllama, batch: list[tuple[str, str, str]], browser_profile: BrowserProfile):
    tasks = [
        check_website(llm, url, name, browser_profile)
        for name, url, _ in batch
    ]
    return asyncio.gather(*tasks, return_exceptions=True)

async def main():
    logger.info("Starting SiteScout website monitoring")
    # Configure browser profile with appropriate settings for website monitoring
    browser_profile = BrowserProfile(
        headless=False,  # Run in headless mode for monitoring
        viewport={"width": 1280, "height": 800},  # Standard desktop viewport
        wait_for_network_idle_page_load_time=3.0,  # Wait longer for slower sites
        maximum_wait_page_load_time=10.0,  # Increase max wait time for slow sites
        wait_between_actions=0.5,  # Reasonable wait between actions
        highlight_elements=False,  # Disable element highlighting for monitoring
        viewport_expansion=0,  # No need for viewport expansion in monitoring
        disable_security=False,  # Keep security enabled
        allowed_domains=None,  # Allow all domains for monitoring
        user_data_dir=None,  # Use ephemeral profile
        storage_state=None,  # No need for persistent storage
        save_recording_path=None,  # Disable recording for monitoring
    )
    try:
        logger.info("Initializing LLM")
        llm = ChatOllama(
            model='qwen2.5:32b-instruct-q4_K_M',
            num_ctx=32000,
            timeout=30
        )
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "categories": {}
        }
        all_checks = []
        for category, websites in WEBSITES.items():
            for name, url in websites.items():
                all_checks.append((name, url, category))
        batch_size = 20
        for i in range(0, len(all_checks), batch_size):
            batch = all_checks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(all_checks) + batch_size - 1)//batch_size}")
            batch_results = await process_batch(llm, batch, browser_profile)
            for (name, url, category), result in zip(batch, batch_results):
                if category not in results["categories"]:
                    results["categories"][category] = {}
                if isinstance(result, Exception):
                    logger.error(f"Error processing website in {category}: {str(result)}")
                    results["categories"][category][name] = {
                        "status": "DOWN",
                        "url": url,
                        "error": str(result)
                    }
                    continue
                name, data = result
                results["categories"][category][name] = data
            if i + batch_size < len(all_checks):
                await asyncio.sleep(1)
    except Exception as e:
        logger.error("Fatal error in main:", exc_info=True)
        raise
    finally:
        logger.info("Saving results to website_status.json")
        with open("website_status.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n=== Summary ===")
        for category, websites in results["categories"].items():
            print(f"\n{category}:")
            for name, data in websites.items():
                status = "✅" if data["status"] == "UP" else "❌"
                print(f"{status} {name}: {data['status']}")
                if data["error"]:
                    print(f"   Error: {data['error']}")
        logger.info("SiteScout monitoring completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error("Fatal error:", exc_info=True)
        sys.exit(1) 