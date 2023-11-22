import random
import nltk
# nltk.download()
nltk.download('punkt')

from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

def synonym_change(question_nl):
    
    # define synonym dictionary
    synonym_dic = {}  
    synonym_dic['user'] = ['user', 'tester', 'person', 'subject']
    synonym_dic['Fridge'] = ['Fridge', 'refrigerator']
    
    question_word_list = word_tokenize(question_nl)

    for idx, word_i in enumerate(question_word_list):
        if word_i in synonym_dic.keys():
            question_word_list[idx] = random.choice(synonym_dic[word_i])
            question_nl_new = TreebankWordDetokenizer().detokenize(question_word_list)
    
    return question_nl_new

# # words in labels
# {'1',
#  '2',
#  '3',
#  'Clean',
#  'Cleanup',
#  'Close',
#  'Coffee',
#  'Cup',
#  'Dishwasher',
#  'Door',
#  'Drawer',
#  'Drink',
#  'Early',
#  'Fridge',
#  'Lie',
#  'Open',
#  'Other',
#  'Relaxing',
#  'Sandwich',
#  'Sit',
#  'Stand',
#  'Switch',
#  'Table',
#  'Toggle',
#  'Walk',
#  'from',
#  'morning',
#  'time'}

# # words in questions
# {"'s",
#  ',',
#  '.',
#  '<',
#  '>',
#  '?',
#  'A1',
#  'A2',
#  'A3',
#  'C1',
#  'Confirm',
#  'Count',
#  'Does',
#  'How',
#  'If',
#  'Is',
#  'Quantify',
#  'R1',
#  'R2',
#  'The',
#  'What',
#  'While',
#  '[',
#  ']',
#  'a',
#  'action',
#  'activity',
#  'after',
#  'and',
#  'be',
#  'before',
#  'by',
#  'can',
#  'case',
#  'characterizes',
#  'correct',
#  'count',
#  'counted',
#  'counts',
#  'do',
#  'does',
#  'doing',
#  'duration',
#  'for',
#  'goes',
#  'he',
#  'how',
#  'if',
#  'in',
#  'instances',
#  'is',
#  'it',
#  'length',
#  'long',
#  'many',
#  'more',
#  'much',
#  'number',
#  'of',
#  'one',
#  'open',
#  'perform',
#  'performing',
#  'performs',
#  'person',
#  'quantify',
#  'quantity',
#  'represents',
#  'right',
#  's',
#  'same',
#  'say',
#  'should',
#  'than',
#  'that',
#  'the',
#  'time',
#  'times',
#  'to',
#  'total',
#  'true',
#  'user',
#  'what',
#  'which',
#  'while',
#  'would',
#  'you',
#  '’'}
