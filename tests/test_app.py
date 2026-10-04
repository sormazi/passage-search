from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_keyword_search_flow():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
    assert not app.exception
    app.radio[0].set_value('Keyword').run()
    app.text_input[0].set_value('backup codes')
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert any('account-guide.md' in x.value for x in app.markdown)
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert any('Upload documents' in x.value for x in app.info)
