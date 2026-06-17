import asyncio
import httpx
import logging
import io
import zipfile
import lxml.etree as etree
from src.services.fda_client import get_past_and_present_labels, SPL_XML_NAMESPACE

logging.basicConfig(level=logging.INFO)

async def test_extraction():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Warfarin is a good candidate for rich label data
        print("Fetching Warfarin labels...")
        past, present = await get_past_and_present_labels("Warfarin", "2015-01-01", client)
        
        # Helper to get XML from label (we can't easily get it from the FDALabelVersion object as it's parsed)
        # So let's manually fetch the zip again for the present label
        from src.services.fda_client import DAILYMED_DOWNLOAD_URL
        
        present_set_id = present.spl_id.split("::v")[0]
        present_version = present.spl_id.split("::v")[1]
        
        params = {"type": "zip", "setid": present_set_id, "version": present_version}
        resp = await client.get(DAILYMED_DOWNLOAD_URL, params=params)
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml_name = [n for n in zf.namelist() if n.endswith(".xml") and not n.startswith("_")][0]
            xml_bytes = zf.read(xml_name)
            
        tree = etree.fromstring(xml_bytes)
        ns = {"hl7": SPL_XML_NAMESPACE}
        codes = tree.xpath("//hl7:section/hl7:code/@code", namespaces=ns)
        display_names = tree.xpath("//hl7:section/hl7:code/@displayName", namespaces=ns)
        
        print("\nStructure of DRUG INTERACTIONS (34073-7) in Present Label:")
        drug_sections = tree.xpath("//hl7:section[hl7:code[@code='34073-7']]", namespaces=ns)
        if drug_sections:
            s = drug_sections[0]
            print(f"Found section. Tag: {s.tag}")
            for child in s:
                print(f"  Child: {child.tag} | Attribute: {child.attrib}")
                if etree.QName(child.tag).localname == "component":
                    inner_section = child.find("hl7:section", namespaces=ns)
                    if inner_section is not None:
                         code_node = inner_section.find("hl7:code", namespaces=ns)
                         if code_node is not None:
                             print(f"    Inner Section Code: {code_node.attrib.get('code')} | Name: {code_node.attrib.get('displayName')}")
        else:
            print("Section 34073-7 not found")


if __name__ == "__main__":
    asyncio.run(test_extraction())
