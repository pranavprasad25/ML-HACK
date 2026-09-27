"""
Verification script for Normalization on real competition samples.
"""
import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import normalize_business_name, normalize_address, normalize_country

def test_samples():
    samples = [
        {
            "name": "Maure Williams Colombier Inc",
            "address": "85 Wayne Avenue, Ticonderoga, NY",
            "country": "US"
        },
        {
            "name": "Maure Wilblims Colombier Inc",
            "address": "",
            "country": "US"
        },
        {
            "name": "Dréxkor",
            "address": "85 Wanye Avenue, Ticonderoga Townshiip, New York",
            "country": "US"
        },
        {
            "name": "B+ Retail Inc",
            "address": "1712 Montebello Avenue, Phoenix, AZ",
            "country": "US"
        },
        {
            "name": "Consulting Nyasa Nursing Private Limited",
            "address": "2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra",
            "country": "India"
        },
        {
            "name": "Dream Construction Limited",
            "address": "H.No.16-11-23/37/A, 2Nd Floor, Flat No.207, Sagar Hotel Building, Opp.Rta Office, Hyderabad, Telangana",
            "country": "India"
        },
        {
            "name": "Café des Artistes S.A.R.L.",
            "address": "12 Rue de la République, 75001 Paris",
            "country": "France"
        }
    ]

    print("=" * 80)
    print("NORMALIZATION PIPELINE TEST")
    print("=" * 80)

    for i, s in enumerate(samples, 1):
        norm_name = normalize_business_name(s["name"])
        norm_addr = normalize_address(s["address"], s["country"])
        norm_country = normalize_country(s["country"])

        print(f"\n--- Sample {i} ---")
        print(f"Original Name:    {s['name']}")
        print(f"Clean Name:       {norm_name['name_clean']}")
        print(f"Base Name:        {norm_name['name_base']}")
        print(f"Legal Suffix:     {norm_name['name_legal_suffix']}")
        print(f"Tokens Sorted:    {norm_name['name_tokens_sorted']}")
        print(f"Original Addr:    {s['address']}")
        print(f"Clean Addr:       {norm_addr['addr_clean']}")
        print(f"Numbers Extracted:{norm_addr['addr_numbers']}")
        print(f"Postal/PIN Code:  {norm_addr['addr_postal_code']}")
        print(f"Has Address?:     {norm_addr['addr_has_val']}")
        print(f"Country:          {norm_country}")

if __name__ == "__main__":
    test_samples()
