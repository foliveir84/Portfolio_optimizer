import requests
import re
import os

def download_infarmed_xls():
    url = "https://extranet.infarmed.pt/CITS-pesquisamedicamento-fo/pesquisaMedicamento.jsf"
    
    session = requests.Session()
    
    # Headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "pt-PT,pt;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    
    print("1. Fetching initial page...")
    response = session.get(url, headers=headers)
    response.raise_for_status()
    
    # Extract ViewState
    # Flexible regex to catch name first, then value, with anything in between
    viewstate_match = re.search(r'name="javax\.faces\.ViewState".*?value="([^"]+)"', response.text)
    
    if not viewstate_match:
        # Try id first
        viewstate_match = re.search(r'id="javax\.faces\.ViewState".*?value="([^"]+)"', response.text)
        
    if not viewstate_match:
        print("Error: Could not find javax.faces.ViewState")
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        return False
        
    view_state = viewstate_match.group(1)
    print(f"   Found ViewState: {view_state[:20]}...")
    
    # Prepare POST data for export
    # These parameters are standard for PrimeFaces/JSF commandButton actions
    payload = {
        "form": "form",
        "form:export-all-button": "",
        "javax.faces.ViewState": view_state
    }
    
    print("2. Requesting export file...")
    # The export usually triggers a file download stream
    post_response = session.post(url, data=payload, headers=headers, stream=True)
    post_response.raise_for_status()
    
    # Check if we got a file or an HTML page (error)
    content_type = post_response.headers.get('Content-Type', '')
    print(f"   Response Content-Type: {content_type}")
    
    if 'text/html' in content_type:
        print("Error: Received HTML instead of file. The button ID or parameters might be wrong.")
        # Debug: Save the HTML to inspect
        with open("debug_error.html", "w", encoding="utf-8") as f:
            f.write(post_response.text)
        return False
        
    filename = "allPackages_python.xls"
    with open(filename, 'wb') as f:
        for chunk in post_response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"Success! Saved to {filename}")
    return True

if __name__ == "__main__":
    download_infarmed_xls()
