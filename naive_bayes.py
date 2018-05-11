import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes
import os

def delete_files(files):
    for f in files:
        os.remove(f)

def find_incompatible_files(path):
    # finds files not compatible with utf-8
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    files = sklearn.datasets.load_files(path)
    num = []
    for i in range(len(files.filenames)):
        try:
            count_vector.fit_transform(files.data[i:i + 1])
        except UnicodeDecodeError:
            num.append(files.filenames[i])
        except ValueError:
            pass
    return num

def bow_vectorizer(files_data):
    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)

def refine_email(email):
    # Delete the unnecessary information in the header of emails

    parts = email.split('\n')
    newparts = []

    finished = False
    for part in parts:
        if finished:
            newparts.append(part)
            continue
        if not (part.startswith('Path:') or part.startswith('Newsgroups:') or part.startswith('Xref:')) and not finished:
            newparts.append(part)
        if part.startswith('Lines:'):
            finished = True
    return '\n'.join(newparts)

def refine_all_emails(file_data):
    for i, email in zip(range(len(file_data)), file_data):
        file_data[i] = refine_email(email)

def remove_files(dir_path):
    # find incompatible files
    incompatible_files = find_incompatible_files(dir_path)
    print str(len(incompatible_files)) + ' utf-8 incompatible files found'

    # delete them
    if(len(incompatible_files) > 0):
        delete_files(incompatible_files)

def test_classifier(X, y, clf, test_size=0.2, y_names=None):
    # train-test split
    print 'test size is: %2.0f%%' % (test_size * 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    print 'Classification report:'
    print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)


def test(path):
    dir_path = path
    remove_files(dir_path)

    # load data
    print 'Loading data'
    files = sklearn.datasets.load_files(dir_path)

    # refine all emails
    refine_all_emails(files.data)

    # calculate the BOW representation
    word_counts = bow_vectorizer(files.data)

    # TFIDF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)

    # create classifier
    clf = sklearn.naive_bayes.MultinomialNB()

    # test the classifier
    test_classifier(X, files.target, clf, test_size=0.2, y_names=['Class 1', 'Class 2'])

# do the main test
test("/Users/chloechen/Downloads/dataset")
