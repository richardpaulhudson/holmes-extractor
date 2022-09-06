import unittest
from packaging import version
import holmes_extractor as holmes
from holmes_extractor.errors import NoDocumentError

holmes_manager = holmes.Manager(
    'en_core_web_trf', perform_coreference_resolution=False, number_of_workers=2)

lg_holmes_manager = holmes.Manager(
    'en_core_web_lg', perform_coreference_resolution=False, number_of_workers=2)

class ManagerTest(unittest.TestCase):

    def _register_multiple_documents_and_search_phrases(self):
        holmes_manager.remove_all_search_phrases()
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets')
        holmes_manager.parse_and_register_document(
            document_text="Everything I know suggests that lions enjoy eating gnu", label='safari')
        holmes_manager.register_search_phrase(
            "A dog chases a cat", label="test")
        holmes_manager.register_search_phrase(
            "A lion eats a gnu", label="test")
        holmes_manager.register_search_phrase(
            "irrelevancy", label="alpha")
        return

    def test_multiple(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match()), 2)

    def test_remove_all_search_phrases(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_all_search_phrases()
        holmes_manager.register_search_phrase("A dog chases a cat")
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_remove_all_documents(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets')
        self.assertEqual(len(holmes_manager.match()), 1)

    def test_remove_all_documents_with_label(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.remove_all_documents()
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets11')
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets12')
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets21')
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets22')
        self.assertEqual(len(holmes_manager.match()), 4)
        holmes_manager.remove_all_documents('pets22')
        self.assertEqual(len(holmes_manager.match()), 3)
        holmes_manager.remove_all_documents('pets1')
        self.assertEqual(len(holmes_manager.match()), 1)
        holmes_manager.remove_all_documents('pets')
        with self.assertRaises(NoDocumentError) as context: 
            holmes_manager.match()

    def test_remove_document(self):
        self._register_multiple_documents_and_search_phrases()
        holmes_manager.parse_and_register_document(
            document_text="All the time I am testing here, dogs keep on chasing cats.", label='pets2')
        self.assertEqual(len(holmes_manager.match()), 3)
        holmes_manager.remove_document(label='pets')
        holmes_manager.remove_document(label='safari')
        matches = holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['document'], 'pets2')

    def test_match_search_phrases_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match(document_text=
            "All the time I am testing here, dogs keep on chasing cats.")), 1)

    def test_match_documents_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match(search_phrase_text=
            "A lion eats a gnu.")), 1)

    def test_match_documents_and_search_phrases_against(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(len(holmes_manager.match(search_phrase_text= "burn",
            document_text="Burn. Everything I know suggests that lions enjoy eating gnu")), 1)
        holmes_manager.remove_all_documents()
        holmes_manager.remove_all_search_phrases()
        self.assertEqual(len(holmes_manager.match(search_phrase_text= "burn",
            document_text="Burn. Everything I know suggests that lions enjoy eating gnu")), 1)

    def test_get_labels(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(holmes_manager.list_search_phrase_labels(),
                         ['alpha', 'test'])

    def test_get_document(self):
        self._register_multiple_documents_and_search_phrases()
        self.assertEqual(holmes_manager.get_document('safari')[5]._.holmes.lemma,
                         'lion')

    def test_remove_all_search_phrases_with_label(self):
        holmes_manager.remove_all_search_phrases()
        holmes_manager.register_search_phrase("testa", label="test1")
        holmes_manager.register_search_phrase("testb", label="test1")
        holmes_manager.register_search_phrase("testc", label="test2")
        holmes_manager.register_search_phrase("testd", label="test2")
        holmes_manager.remove_all_search_phrases_with_label("test2")
        holmes_manager.remove_all_search_phrases_with_label("testb")
        self.assertEqual(holmes_manager.list_search_phrase_labels(),
                         ['test1'])
        self.assertEqual(len(holmes_manager.match(document_text=
            "testa")), 1)
        self.assertEqual(len(holmes_manager.match(document_text=
            "testb")), 1)
        self.assertEqual(len(holmes_manager.match(document_text=
            "testc")), 0)
        self.assertEqual(len(holmes_manager.match(document_text=
            "testd")), 0)

    def test_pipe_with_single_process(self):
        docs = lg_holmes_manager.nlp.pipe(['document1', 'document2'])
        self.assertEqual(str(next(docs)), 'document1')
        

    def test_pipe_with_multiple_processes(self):
        docs = lg_holmes_manager.nlp.pipe(['document1', 'document2'], n_process=2)
        self.assertEqual(str(next(docs)), 'document1')
        self.assertEqual(str(next(docs)), 'document2')

    @unittest.skipIf(version.parse(holmes_manager.nlp.meta["version"]) < version.parse("3.3.0"), 'spaCy feature not supported')
    def test_bespoke_entity_labels(self):
        tmp_holmes_manager = holmes.Manager("en_core_web_lg", entity_labels_to_corresponding_lexemes={"SQUIRREL": "squirrel"}, overall_similarity_threshold=0.99)
        config = {"spans_key": None, "annotate_ents": True, "overwrite": False}
        ruler = tmp_holmes_manager.nlp.add_pipe("span_ruler", config=config)
        patterns = [{"label": "SQUIRREL", "pattern": "ice cream"}]
        ruler.add_patterns(patterns)
        tmp_holmes_manager.parse_and_register_document("We ate some ice cream")
        tmp_holmes_manager.register_search_phrase("Somebody eats squirrels")
        matches = tmp_holmes_manager.match()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["word_matches"][-1]["explanation"], "Has an entity label that is 100% similar to the word embedding corresponding to SQUIRREL.")


    def test_retokenization(self):
        doc = holmes_manager.nlp("This is a test")
        with doc.retokenize() as retokenizer:
            retokenizer.merge(doc[0:4], {'POS': 'NOUN'})
        coreferee_ext = holmes_manager.nlp.get_pipe("coreferee")
        holmes_ext = holmes_manager.nlp.get_pipe("holmes")
        coreferee_ext(doc)
        holmes_ext(doc)
        holmes_manager.remove_all_documents()
        holmes_manager.register_serialized_document(doc.to_bytes(), 'test')
        matches = holmes_manager.match(search_phrase_text="ENTITYNOUN")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["word_matches"][0]["document_word"], "this is a test")