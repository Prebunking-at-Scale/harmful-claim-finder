"""
This demo script runs the transcript inference script, which pipes keywords into PASTEL for a transcript.
"""

from pprint import pp

from harmful_claim_finder.transcript_inference import run_checkworthy

sentences = [
    "But in words that could further enrage his critics, Starmer insisted that new migrants must “learn the language and integrate” once in the UK.",
    "He said: “Britain is an inclusive and tolerant country, but the public expect that people who come here should be expected to learn the language and integrate.”",
    "Net migration, the difference between the number of people moving to the UK and the number leaving, was 728,000 in the 12 months to June 2024.",
    "Hamas has released Edan Alexander, a Israeli American hostage held in Gaza who was taken captive while serving in the Israeli army during Hamas's attack on 7 October.",
    "Israel claimed 20 of its warplanes on Monday had completely destroyed the Houthi-held port of Hodeidah, as well as a nearby cement factory.",
    "Israel was shocked that its air defences were penetrated by a single Houthi missile, but Iran believes Israel is escalating the crisis in an attempt to disrupt the negotiations between the US and Iran over its nuclear programme.",
    "Giving obese children weight loss jabs works and could help avoid arguments over mealtimes, according to research.",
    "The clinicians found that nearly a third of these children dropped enough weight to improve their health, compared with about 27% in earlier treated groups with no access to the drugs.",
    "Manchester United intend to retain Ruben Amorim as head coach next season even if they lose the Europa League final to Tottenham.",
]

kw = {
    "health": ["doctor", "health", "hospital", "nurses", "obesity", "medicine"],
    "war": ["gun", "bomb", "war", "warplanes", "army", "strike", "attack"],
    "migration": ["crossings", "migrants", "immigration", "migration"],
}

countries = ["GBR", "USA"]

result = run_checkworthy(kw, sentences, countries)

pp(result)
