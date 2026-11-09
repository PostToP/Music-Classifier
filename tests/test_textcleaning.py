import json
import tempfile
import unittest
from pathlib import Path

from data.text_cleaning import TextPreprocessor
from model.model_wrapper import encode_tfidf
from vectorizer.vectorizer_tfidf import VectorizerTFIDF


class TextCleaningTest(unittest.TestCase):
    def test_category_extraction(self):
        self.assertEqual(TextPreprocessor.clean_categories(["https://en.wikipedia.org/wiki/Music"]), ["Music"])

    def test_duration_conversion(self):
        self.assertEqual(TextPreprocessor.convert_duration("PT2M3S"), 123)
        self.assertEqual(TextPreprocessor.convert_duration("PT1H2M3S"), 3723)

    def test_escaped_characters(self):
        self.assertEqual(
            TextPreprocessor.remove_escaped_characters("This is a test string with no escaped characters"),
            "This is a test string with no escaped characters",
        )
        self.assertEqual(
            TextPreprocessor.remove_escaped_characters("This is a\n test\tstring\rwith escaped\n characters"),
            "This is a  test string with escaped  characters",
        )

    def test_remove_fancy(self):
        self.assertEqual(TextPreprocessor.normalize_text_to_ascii("𝑀𝓎 𝒷𝑒𝓈𝓉 𝒶𝒸𝒸"), "My best acc")
        self.assertEqual(TextPreprocessor.normalize_text_to_ascii("𝑪𝒐𝒎𝒆, 𝒃𝒓𝒆𝒂𝒌 𝒎𝒆 𝒅𝒐𝒘𝒏"), "Come, break me down")
        self.assertEqual(TextPreprocessor.normalize_text_to_ascii("𝘽𝙀𝙂𝙂𝙄𝙉"), "BEGGIN")
        self.assertEqual(TextPreprocessor.normalize_text_to_ascii("𝕮𝖔𝖑𝖑𝖆𝖗 𝖔𝖋 𝕿𝖗𝖚𝖙𝖍"), "Collar of Truth")
        self.assertEqual(
            TextPreprocessor.normalize_text_to_ascii("𝗜𝗻𝗮 𝗮𝗱𝗱𝗿𝗲𝘀𝘀𝗲𝘀 𝗵𝗲𝗿 𝘂𝗻𝗳𝗼𝗿𝘁𝘂𝗻𝗮𝘁𝗲 𝘀𝗶𝘁𝘂𝗮𝘁𝗶𝗼𝗻"),
            "Ina addresses her unfortunate situation",
        )
        self.assertEqual(
            TextPreprocessor.normalize_text_to_ascii("𝓢𝓗𝓔 𝓖𝓞𝓣 𝓜𝓔 𝓕𝓔𝓔𝓛𝓘𝓝𝓖 𝓛𝓘𝓚𝓔 𝓐𝓝 𝓐𝓝𝓖𝓔𝓛"),
            "SHE GOT ME FEELING LIKE AN ANGEL",
        )
        # self.assertEqual(TextPreprocessor.normalize_text_to_ascii("₴ⱧɆ ĐØ₦₮ ₩₳₦₮ ₥Ɇ Ⱡ₳₮ɆⱠɎ"), "SHE DONT WANT ME LATELY")
        self.assertEqual(TextPreprocessor.normalize_text_to_ascii("𝓝𝓝𝓐𝓐𝓐𝓓"), "NNAAAD")
        self.assertEqual(TextPreprocessor.normalize_text_to_ascii("𝔖𝔓ℑℜℑ𝔗 𝔒𝔉 𝔏ℑ𝔈"), "SPIRIT OF LIE")
        # self.assertEqual(TextPreprocessor.normalize_text_to_ascii("ιм α ℓιттℓє ѕну тυρє"), "im a little shy type")
        self.assertEqual(
            TextPreprocessor.normalize_text_to_ascii("𝗥𝗘𝗗 𝗜𝗦 𝗧𝗛𝗘 𝗖𝗢𝗟𝗢𝗨𝗥 𝗢𝗙 𝗠𝗬 𝗟𝗢𝗩𝗘"),
            "RED IS THE COLOUR OF MY LOVE",
        )

    def test_link_conversion(self):
        self.assertEqual(TextPreprocessor.replace_links_with_sld("https://www.google.com"), " google ")
        self.assertEqual(TextPreprocessor.replace_links_with_sld("www.google.com"), " google ")
        self.assertEqual(TextPreprocessor.replace_links_with_sld("https://www.youtube.com"), " youtube ")
        self.assertEqual(
            TextPreprocessor.replace_links_with_sld("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            " youtube ",
        )
        self.assertEqual(
            TextPreprocessor.replace_links_with_sld(
                "https:// www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley",
            ),
            " youtube ",
        )
        self.assertEqual(TextPreprocessor.replace_links_with_sld("http://subdomain.example.co.uk"), " example ")
        self.assertEqual(TextPreprocessor.replace_links_with_sld("https://www.youtube.com/Gamebreaker64"), " youtube ")

    def test_email(self):
        self.assertEqual(TextPreprocessor.remove_emails("test@example.com"), " ")
        self.assertEqual(
            TextPreprocessor.remove_emails("This is a test email: test@example.com"),
            "This is a test email:  ",
        )

    def test_remove_ats(self):
        self.assertEqual(TextPreprocessor.remove_ats("@username"), " ")
        self.assertEqual(TextPreprocessor.remove_ats("Mentioning @someone"), "Mentioning  ")
        self.assertEqual(TextPreprocessor.remove_ats("Hey, Hey, Hey. @you"), "Hey, Hey, Hey.  ")

    def test_to_lower(self):
        self.assertEqual(TextPreprocessor.to_lower("This is a TEST string."), "this is a test string.")
        self.assertEqual(TextPreprocessor.to_lower("ANOTHER Test"), "another test")

    def test_expand_contractions(self):
        self.assertEqual(TextPreprocessor.expand_contractions("I'm not sure."), "I am not sure.")
        self.assertEqual(TextPreprocessor.expand_contractions("You're awesome!"), "You are awesome!")
        self.assertEqual(TextPreprocessor.expand_contractions("It's a beautiful day."), "It is a beautiful day.")
        self.assertEqual(TextPreprocessor.expand_contractions("Can't wait to see you."), "Cannot wait to see you.")
        self.assertEqual(TextPreprocessor.expand_contractions("I've been there before."), "I have been there before.")
        self.assertEqual(TextPreprocessor.expand_contractions("She'd love to go."), "She would love to go.")
        self.assertEqual(TextPreprocessor.expand_contractions("They'll be here soon."), "They will be here soon.")
        self.assertEqual(
            TextPreprocessor.expand_contractions("We're going to the movies."),
            "We are going to the movies.",
        )
        self.assertEqual(TextPreprocessor.expand_contractions("Don't forget your keys."), "Do not forget your keys.")
        self.assertEqual(
            TextPreprocessor.expand_contractions("Shouldn't you be working?"),
            "Should not you be working?",
        )
        self.assertEqual(
            TextPreprocessor.expand_contractions("Neverland feat. Hatsune Miku"),
            "Neverland featuring. Hatsune Miku",
        )

    def test_remove_hypens(self):
        self.assertEqual(TextPreprocessor.remove_hyphen("This is a test-string."), "This is a teststring.")
        self.assertEqual(TextPreprocessor.remove_hyphen("This is a test-string-test."), "This is a teststringtest.")
        self.assertEqual(
            TextPreprocessor.remove_hyphen("This is a test-string-test-string."),
            "This is a teststringteststring.",
        )

    def test_remove_special_chars(self):
        self.assertEqual(
            TextPreprocessor.remove_special_characters("This is a test string with special characters like !@#$%^&*()"),
            "This is a test string with special characters like           ",
        )
        self.assertEqual(TextPreprocessor.remove_special_characters("This is another test."), "This is another test ")
        self.assertEqual(TextPreprocessor.remove_special_characters("12345"), "     ")
        self.assertEqual(
            TextPreprocessor.remove_special_characters("Test string with numbers 123"),
            "Test string with numbers    ",
        )

    def test_collapse_whitespace(self):
        self.assertEqual(
            TextPreprocessor.collapse_whitespaces("This is a   test string  with   multiple   spaces."),
            "This is a test string with multiple spaces.",
        )
        self.assertEqual(
            TextPreprocessor.collapse_whitespaces("  Leading and trailing spaces  "),
            "Leading and trailing spaces",
        )
        self.assertEqual(TextPreprocessor.collapse_whitespaces("No extra spaces here"), "No extra spaces here")

    def test_lemmatize_as_nouns(self):
        self.assertEqual(TextPreprocessor.lemmatize_text("cats"), "cat")
        self.assertEqual(TextPreprocessor.lemmatize_text("songs"), "song")
        self.assertEqual(
            TextPreprocessor.lemmatize_text("The quick brown foxes are jumping"),
            "The quick brown fox are jumping",
        )


class TfidfEncodingTest(unittest.TestCase):
    def test_prod_tfidf_encoding_matches_training(self):
        vectorizer = VectorizerTFIDF(10)
        vectorizer.train([["music", "video", "music"], ["live", "video"], ["official", "music", "audio"]])
        vocab = vectorizer.vectorizer.vocabulary_
        idf = vectorizer.vectorizer.idf_.astype("float32")
        tokens = ["music", "music", "video", "neverseen"]
        expected_vector = vectorizer.encode(tokens).round(6).tolist()
        self.assertEqual(encode_tfidf(tokens, vocab, idf).round(6).tolist(), expected_vector)

    def test_excluded_tokens_stay_out_of_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            excluded_tokens_path = Path(directory) / "artist_tokens.json"
            excluded_tokens_path.write_text(json.dumps(["miku"]))
            vectorizer = VectorizerTFIDF(10, excluded_tokens_path=str(excluded_tokens_path))
            vectorizer.train([["miku", "song"]])
        self.assertNotIn("miku", vectorizer.vectorizer.vocabulary_)
        self.assertEqual(vectorizer.encode(["miku", "song"]).tolist(), [1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
