"""Document parsing tools (basic version)"""

def parse_text_input(text: str) -> dict:
    """Parse financial data from text input"""
    # Simple parser - can be enhanced with NLP
    data = {
        "income": None,
        "expenses": None,
        "found": False
    }
    
    # Simple keyword extraction
    text_lower = text.lower()
    
    # Try to find income
    if "salary" in text_lower or "income" in text_lower:
        # Extract numbers (simple approach)
        import re
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        if numbers:
            data["income"] = float(numbers[0].replace(',', ''))
            data["found"] = True
    
    # Try to find expenses
    if "expense" in text_lower or "spend" in text_lower:
        import re
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        if len(numbers) > 1:
            data["expenses"] = float(numbers[1].replace(',', ''))
        elif len(numbers) == 1 and data["income"] is None:
            data["expenses"] = float(numbers[0].replace(',', ''))
        data["found"] = True
    
    return data

# For PDF parsing, you'd need: pip install PyPDF2
# def extract_from_pdf(pdf_file):
#     # Implementation for PDF extraction
#     pass