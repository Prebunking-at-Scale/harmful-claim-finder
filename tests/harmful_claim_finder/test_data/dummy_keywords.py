keywords_type = dict[str, dict[str, dict[str, dict[str, list[str]]]]]
test_keywords: keywords_type = {
    "include_lists": {
        "fullfact": {
            "asylum": {
                "en": [
                    "Small boat crossings",
                    "Small boats",
                    "Rwanda policy",
                    "Visa applications",
                    "Net migration",
                    "Illegal migrants",
                    "Asylum seekers",
                    "Forced returns",
                    "asylum hotels",
                    "red days",
                ]
            },
            "crime": {
                "en": [
                    "Police officers",
                    "Knife crime",
                    # "Sadiq Khan",
                ]
            },
            "culture-wars": {
                "en": [
                    "ULEZ",
                    "Gender ideology",
                    "Gender-neutral",
                    "Conversion therapy",
                ]
            },
            "defence": {
                "en": [
                    "Napoleonic wars",
                    "Defence spending",
                    "Armed forces",
                    "2.5% of GDP",
                    "strategic defence review",
                    "path to spending 2.5% of GDP on defence",
                ]
            },
            "economy": {
                "en": [
                    "Gross domestic product",
                    "GDP",
                    "Triple lock",
                    "Disposable income",
                    "Working people",
                    "National Insurance",
                    "National Insurance Contributions",
                    "Debt burden",
                    "Income tax",
                    "National Debt",
                    "Tax burden",
                    "Council tax",
                    "Tax bills",
                    "GDP growth",
                    "Tax cuts",
                    "Tax rises",
                    "22 billion shortfall",
                    "22 billion black hole",
                    "Employer contributions",
                    "The Budget",
                    "Inheritance tax ",
                    "spending review",
                ]
            },
            "education": {
                "en": [
                    "Ghost children",
                    "Private school",
                    "VAT private school",
                    "private schools bill ",
                ]
            },
            "energy": {
                "en": [
                    "Energy bills",
                    "Nuclear plants",
                    "Green levies",
                    "Gas licences",
                    "Wind farms",
                    "Net zero",
                    "Great British Energy",
                    "Energy price cap",
                    "Windfall tax",
                    "fuel poverty",
                    "carbon capture",
                ]
            },
            "general": {
                "en": [
                    "Record high",
                    "record numbers",
                    "Since WWII",
                    "Since World War",
                    "Since 2019",
                    "Under the Conservatives",
                    "since Labour",
                ]
            },
            "health": {
                "en": [
                    "NHS Strikes",
                    "Waiting lists",
                    "New hospitals",
                    "Covid-19 Pandemic",
                    "18-week",
                ]
            },
            "housing": {
                "en": [
                    "Social housing",
                    "Affordable housing",
                    "Council housing",
                    "homelessness",
                    "renters rights bill",
                    "planning and infrastructure bill ",
                    "150 major infrastructure projects",
                ]
            },
            "poverty": {
                "en": ["Child poverty", "Food parcels", "Food banks", "benefit cuts "]
            },
            "transport": {
                "en": [
                    "HS2",
                    "Great British Railways",
                    "Train drivers",
                    "potholes",
                    "renationalisation ",
                ]
            },
            "topical": {
                "en": [
                    "Kemi Badenoch",
                    "Donald Trump",
                    "assisted dying bill",
                    "football governance bill ",
                    "Plan for Change",
                    "grooming gangs",
                    "council elections",
                    " two-tier councils",
                    "Ukraine aid",
                    "Early release",
                    "Prison overcrowding",
                    "spending review",
                    "steel tarrifs ",
                    "nhs cuts",
                    "civil service cuts",
                ]
            },
        }
    },
    "exclude_lists": {"fullfact": {"education": {"en": []}}},
}
