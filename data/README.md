## Dataset
We provide three processed datasets: Amazon-book and Yelp2018.
* You can find the full version of recommendation datasets via [Amazon-book](http://jmcauley.ucsd.edu/data/amazon) and [Yelp2018](https://www.yelp.com/dataset/challenge).
* We follow [KB4Rec](https://github.com/RUCDM/KB4Rec) to preprocess Amazon-book and Last-FM datasets, mapping items into Freebase entities via title matching if there is a mapping available.

| | | Amazon-book | Last-FM | Yelp2018 |
|:---:|:---|---:|---:|---:|
|User-Item Interaction| #Users | 70,679 | 23,566 | 45,919|
| | #Items | 24,915 | 48,123 | 45,538|
| | #Interactions | 847,733 | 3,034,796 | 1,185,068|
|Knowledge Graph | #Entities | 88,572 | 58,266 | 90,961|
| | #Relations | 39 | 9 | 42 |
| | #Triplets | 2,557,746 | 464,567 | 1,853,704|


* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
  
* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (`org_id`, `remap_id`) for one user, where `org_id` and `remap_id` represent the ID of such user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (`org_id`, `remap_id`, `freebase_id`) for one item, where `org_id`, `remap_id`, and `freebase_id` represent the ID of such item in the original, our datasets, and freebase, respectively.
  
* `entity_list.txt`
  * Entity file.
  * Each line is a triplet (`freebase_id`, `remap_id`) for one entity in knowledge graph, where `freebase_id` and `remap_id` represent the ID of such entity in freebase and our datasets, respectively.
  
* `relation_list.txt`
  * Relation file.
  * Each line is a triplet (`freebase_id`, `remap_id`) for one relation in knowledge graph, where `freebase_id` and `remap_id` represent the ID of such relation in freebase and our datasets, respectively.
  