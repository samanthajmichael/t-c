# Import main RAG functionality
from .rag import (
    initialize_rag,
    create_retriever,
    encode_documents,
    setup_environment
)

# Import helper functions
from .helpers import (
    text_wrap,
    show_context,
    encode_pdf,
    encode_from_string,
    retrieve_context_per_question,
    read_pdf_to_string,
    bm25_retrieval,
    QuestionAnswerFromContext,
    create_question_answer_from_context_chain,
    answer_question_from_context
)

# Make these functions available when importing from core
__all__ = [
    # RAG functions
    'initialize_rag',
    'create_retriever',
    'encode_documents',
    'setup_environment',
    
    # Helper functions
    'text_wrap',
    'show_context',
    'encode_pdf',
    'encode_from_string',
    'retrieve_context_per_question',
    'read_pdf_to_string',
    'bm25_retrieval',
    'QuestionAnswerFromContext',
    'create_question_answer_from_context_chain',
    'answer_question_from_context'
]