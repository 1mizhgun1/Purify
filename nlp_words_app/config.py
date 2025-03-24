import pandas as pd
import nltk
import pymorphy3
from spell_checker import SpellChecker
nltk.download('stopwords')
from pathlib import Path

BASE_DIR = Path(__file__).parent

ORFO_DATASET_PATH = BASE_DIR / "data" / "orfo_and_typos.L1_5+PHON.csv"
KARTASLOVSENT_PATH = BASE_DIR / "data" / "kartaslovsent.csv"


mat_regex = r"""(?iux)(?<![а-яё])(?:
(?:(?:у|[нз]а|(?:хитро|не)?вз?[ыьъ]|с[ьъ]|(?:и|ра)[зс]ъ?|(?:о[тб]|п[оа]д)[ьъ]?|(?:\S(?=[а-яё]))+?[оаеи-])-?)?(?:
  [её](?:б(?!о[рй]|рач)|п[уа](?:ц|тс))|
  и[пб][ае][тцд][ьъ]
).*?|

(?:(?:н[иеа]|(?:ра|и)[зс]|[зд]?[ао](?:т|дн[оа])?|с(?:м[еи])?|а[пб]ч|в[ъы]?|пр[еи])-?)?ху(?:[яйиеёю]|л+и(?!ган)).*?|

бл(?:[эя]|еа?)(?:[дт][ьъ]?)?|

\S*?(?:
  п(?:
    [иеё]зд|
    ид[аое]?р|
    ед(?:р(?!о)|[аое]р|ик)|
    охую
  )|
  бля(?:[дбц]|тс)|
  [ое]ху[яйиеё]|
  хуйн
).*?|

(?:о[тб]?|про|на|вы)?м(?:
  анд(?:[ауеыи](?:л(?:и[сзщ])?[ауеиы])?|ой|[ао]в.*?|юк(?:ов|[ауи])?|е[нт]ь|ища)|
  уд(?:[яаиое].+?|е?н(?:[ьюия]|ей))|
  [ао]л[ао]ф[ьъ](?:[яиюе]|[еёо]й)
)|

елд[ауые].*?|
ля[тд]ь|
(?:[нз]а|по)х
)(?![а-яё])"""

PRONOUNS = ['я', 'ты', 'вы', 'он', 'она', 'оно', 'мы', 'они', 'вас', 'нас', 'их', 'его', 'её']
stopword_set = set(nltk.corpus.stopwords.words('russian'))
stopword_set = stopword_set.union({'это', 'который', 'весь', 'наш', 'свой', 'ещё', 'её', 'ваш', 'также', 'итак'})

words_semantics_dataset = pd.read_csv(KARTASLOVSENT_PATH, sep=";")
term_to_tag_dict = dict(zip(words_semantics_dataset['term'], words_semantics_dataset['tag']))

lemmatizer = pymorphy3.MorphAnalyzer()

checker = SpellChecker(ORFO_DATASET_PATH)

