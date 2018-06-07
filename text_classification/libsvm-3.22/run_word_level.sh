./svm-train -s 0 -c 5 -t 0 -g 0.5 -e 0.1 \
  ../cnews_data/word-level-feature/cnews.train.txt

./svm-predict \
  ../cnews_data/word-level-feature/cnews.train.txt \
  cnews.train.txt.model \
  ../cnews_data/word-level-feature/cnews.train.txt.result

./svm-predict \
  ../cnews_data/word-level-feature/cnews.val.txt \
  cnews.train.txt.model \
  ../cnews_data/word-level-feature/cnews.val.txt.result

./svm-predict \
  ../cnews_data/word-level-feature/cnews.test.txt \
  cnews.train.txt.model \
  ../cnews_data/word-level-feature/cnews.test.txt.result

