import os
import unittest
from unittest.mock import MagicMock, patch
from rag_backend import CareerMentorRAG

class TestCareerMentorRAG(unittest.TestCase):
    
    @patch('rag_backend.GoogleGenerativeAIEmbeddings')
    @patch('rag_backend.ChatGoogleGenerativeAI')
    @patch('rag_backend.Chroma')
    def test_init_gemini(self, mock_chroma, mock_llm, mock_embeddings):
        """Test initialization of Gemini configuration."""
        mentor = CareerMentorRAG("Google Gemini", "fake_gemini_key")
        self.assertEqual(mentor.provider, "Google Gemini")
        self.assertEqual(mentor.api_key, "fake_gemini_key")
        mock_embeddings.assert_called_once_with(model="models/gemini-embedding-001", google_api_key="fake_gemini_key")
        mock_llm.assert_called_once_with(model="gemini-2.5-flash", temperature=0.3, google_api_key="fake_gemini_key")
        mock_chroma.assert_called_once()

    @patch('rag_backend.OpenAIEmbeddings')
    @patch('rag_backend.ChatOpenAI')
    @patch('rag_backend.Chroma')
    def test_init_openai(self, mock_chroma, mock_llm, mock_embeddings):
        """Test initialization of OpenAI configuration."""
        mentor = CareerMentorRAG("OpenAI", "fake_openai_key")
        self.assertEqual(mentor.provider, "OpenAI")
        self.assertEqual(mentor.api_key, "fake_openai_key")
        mock_embeddings.assert_called_once_with(model="text-embedding-3-small", openai_api_key="fake_openai_key")
        mock_llm.assert_called_once_with(model="gpt-4o-mini", temperature=0.3, openai_api_key="fake_openai_key")
        mock_chroma.assert_called_once()

    @patch('rag_backend.GoogleGenerativeAIEmbeddings')
    @patch('rag_backend.ChatGoogleGenerativeAI')
    @patch('rag_backend.Chroma')
    def test_parse_file_txt(self, mock_chroma, mock_llm, mock_embeddings):
        """Test plain text parser."""
        mentor = CareerMentorRAG("Google Gemini", "fake_key")
        sample_txt = b"Hello, this is a test document."
        parsed = mentor.parse_file(sample_txt, "resume.txt", "text/plain")
        self.assertEqual(parsed, "Hello, this is a test document.")

    @patch('rag_backend.GoogleGenerativeAIEmbeddings')
    @patch('rag_backend.ChatGoogleGenerativeAI')
    @patch('rag_backend.Chroma')
    def test_add_document(self, mock_chroma, mock_llm, mock_embeddings):
        """Test adding document, splitting it, and saving it to the vector store."""
        mock_vs_instance = MagicMock()
        mock_chroma.return_value = mock_vs_instance
        
        mentor = CareerMentorRAG("Google Gemini", "fake_key")
        file_content = b"This is a sample resume text. " * 50
        
        success = mentor.add_document(file_content, "sample_resume.txt", "text/plain", "Resume/Tips")
        
        self.assertTrue(success)
        mock_vs_instance.add_documents.assert_called_once()
        
        # Verify document metadata mapping
        added_docs = mock_vs_instance.add_documents.call_args[0][0]
        self.assertGreater(len(added_docs), 0)
        self.assertEqual(added_docs[0].metadata["source"], "sample_resume.txt")
        self.assertEqual(added_docs[0].metadata["category"], "Resume/Tips")

if __name__ == '__main__':
    unittest.main()
