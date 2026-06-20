import pytest
from datetime import datetime
import lxml.etree as etree

# Import the functions from your main module (assuming it's named fda_client.py)
from src.services.fda_client import (
    _select_active_modern_set_id,
    _select_deepest_historical_set_id,
    _parse_dailymed_date,
    _extract_section_text,
    SPL_XML_NAMESPACE
)

def test_parse_dailymed_date():
    """Test the locale-independent date parser handles DailyMed strings."""
    assert _parse_dailymed_date("Sep 26, 2012") == datetime(2012, 9, 26)
    assert _parse_dailymed_date("Jan 01, 2020") == datetime(2020, 1, 1)
    
    with pytest.raises(ValueError):
        _parse_dailymed_date("Invalid Date String")

def test_select_active_modern_set_id():
    """
    Test that the modern selector correctly picks the labeler with the 
    highest spl_version, falling back to published_date on a tie.
    """
    candidates = [
        # Candidate A: Version 2
        {"setid": "uuid-A", "spl_version": "2", "published_date": "Jan 01, 2020"},
        
        # Candidate B: Version 5, older date (Should lose tie-breaker to C)
        {"setid": "uuid-B", "spl_version": "5", "published_date": "Feb 01, 2019"},
        
        # Candidate C: Version 5, newer date (Should WIN)
        {"setid": "uuid-C", "spl_version": "5", "published_date": "Mar 01, 2021"},
        
        # Candidate D: Missing version data (Should be handled gracefully)
        {"setid": "uuid-D"}
    ]
    
    winner = _select_active_modern_set_id(candidates)
    assert winner == "uuid-C", "Failed to select the modern ID based on version/date waterfall"


@pytest.mark.asyncio
async def test_select_deepest_historical_set_id_empty_raises():
    """Ensure _select_deepest_historical_set_id raises on an empty candidate list."""
    import httpx

    rx_date = datetime(2020, 1, 1)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="No candidates provided"):
            await _select_deepest_historical_set_id([], client, rx_date)

def test_extract_section_text_with_tables_and_lists():
    """
    Test the XML extractor to ensure it doesn't drop tables and lists, 
    and formats them correctly.
    """
    loinc_code = "34066-1" # Generic LOINC for Boxed Warning
    
    # Mock SPL XML with a paragraph, a list, and a table
    mock_xml = f"""
    <document xmlns:hl7="{SPL_XML_NAMESPACE}">
        <hl7:component>
            <hl7:section>
                <hl7:code code="{loinc_code}"/>
                <hl7:text>
                    <hl7:paragraph>This drug carries serious risks.</hl7:paragraph>
                    <hl7:list>
                        <hl7:item>Risk of bleeding</hl7:item>
                        <hl7:item>Risk of fainting</hl7:item>
                    </hl7:list>
                    <hl7:table>
                        <hl7:tbody>
                            <hl7:tr>
                                <hl7:td>Dose</hl7:td>
                                <hl7:td>Effect</hl7:td>
                            </hl7:tr>
                            <hl7:tr>
                                <hl7:td>10mg</hl7:td>
                                <hl7:td>High</hl7:td>
                            </hl7:tr>
                        </hl7:tbody>
                    </hl7:table>
                </hl7:text>
            </hl7:section>
        </hl7:component>
    </document>
    """
    
    tree = etree.fromstring(mock_xml.encode("utf-8"))
    extracted_text = _extract_section_text(tree, loinc_code)

    assert extracted_text is not None, "Section extraction returned None"
    
    # Assertions to ensure our formatting logic worked
    assert "This drug carries serious risks." in extracted_text
    assert "• Risk of bleeding" in extracted_text
    assert "• Risk of fainting" in extracted_text
    assert "Dose | Effect" in extracted_text
    assert "10mg | High" in extracted_text