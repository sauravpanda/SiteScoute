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

async def check_website(browser_session: BrowserSession, llm: ChatOllama, url: str, name: str) -> tuple[str, dict]:
    """Check website status using browser-use agent"""
    logger.info(f"Checking website: {name} ({url})")
    
    task = f"""Visit {url} and check if the website is working properly.
    Simply respond with a JSON:
    {{
        "status": "UP" if the page loads successfully, "DOWN" if it fails,
        "reason": "brief explanation of what you see"
    }}
    """
    
    try:
        # Create and run agent for this website
        agent = Agent(
            task=task,
            browser_session=browser_session,
            llm=llm
        )
        
        result = await agent.run()
        
        # Extract the last message from the agent's history
        last_message = result.messages[-1].content if result.messages else ""
        logger.debug(f"Agent response for {name}: {last_message[:200]}...")
        
        try:
            # Try to parse the JSON response
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
            
    except Exception as e:
        logger.error(f"Error checking {name}: {str(e)}", exc_info=True)
        return name, {
            "status": "DOWN",
            "url": url,
            "error": str(e)
        }

async def main():
    logger.info("Starting SiteScout website monitoring")
    
    # Initialize browser session with recommended settings
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            disable_security=True,
            headless=False,
            save_recording_path='./tmp/recordings',
            user_data_dir='~/.config/browseruse/profiles/default',
        ),
        keep_alive=True,
        max_tabs=20  # Increased for better parallel processing
    )
    
    try:
        logger.info("Starting browser session")
        await browser_session.start()
        
        # Initialize LLM
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
        
        # Process all websites in parallel
        all_tasks = []
        for category, websites in WEBSITES.items():
            logger.info(f"Creating tasks for category: {category}")
            category_tasks = [
                check_website(browser_session, llm, url, name)
                for name, url in websites.items()
            ]
            all_tasks.extend(category_tasks)
        
        # Run all website checks in parallel
        logger.info(f"Running {len(all_tasks)} website checks in parallel")
        website_results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Organize results by category
        current_category = None
        for (category, websites), result in zip(WEBSITES.items(), website_results):
            if category != current_category:
                current_category = category
                results["categories"][category] = {}
            
            if isinstance(result, Exception):
                logger.error(f"Error processing website in {category}: {str(result)}")
                continue
                
            name, data = result
            results["categories"][category][name] = data
            
    except Exception as e:
        logger.error("Fatal error in main:", exc_info=True)
        raise
    finally:
        # Save results to a JSON file
        logger.info("Saving results to website_status.json")
        with open("website_status.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n=== Summary ===")
        for category, websites in results["categories"].items():
            print(f"\n{category}:")
            for name, data in websites.items():
                status = "✅" if data["status"] == "UP" else "❌"
                print(f"{status} {name}: {data['status']}")
                if data["error"]:
                    print(f"   Error: {data['error']}")
        
        # Close browser session
        logger.info("Closing browser session")
        try:
            await browser_session.close()
        except Exception as e:
            logger.error(f"Error closing browser session: {str(e)}")
        logger.info("SiteScout monitoring completed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error("Fatal error:", exc_info=True)
        sys.exit(1) 