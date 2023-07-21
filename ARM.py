import time
import psutil
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt


def get_filtered_rules(rules, desired_items_lhs_size):
    desired_items_rhs = {'cardio_0', 'cardio_1'}
    filtered_rules = []
    for _, rule in rules.iterrows():
        antecedents = rule['antecedents']
        consequents = rule['consequents']
        support = rule['support']
        confidence = rule['confidence']
        lift = rule['lift']
        antecedents_str = ', '.join(str(item) for item in antecedents)
        consequents_str = ', '.join(str(item) for item in consequents)
        if len(consequents) == 1 and consequents_str in desired_items_rhs:
            if len(antecedents) >= desired_items_lhs_size:
                rule_str = f"{{{antecedents_str}}} -> {{{consequents_str}}} (support: {support}, confidence: {confidence}, lift: {lift})"
                filtered_rules.append(rule_str)

    return filtered_rules


def apriori_algo(data, min_support, min_confidence):
    process = psutil.Process()
    before_memory = process.memory_info().rss / 1024 / 1024  # Memory usage before the algorithm in MB
    itemsets = apriori(data.astype('bool'), min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric='confidence', min_threshold=min_confidence)
    after_memory = process.memory_info().rss / 1024 / 1024  # Memory usage before the algorithm in MB
    used_memory = after_memory - before_memory
    return itemsets, rules


df_binary = pd.read_csv(r'C:\Users\flori\PycharmProjects\pythonProject\preprocessed_file_arm.csv', sep=",")
df_binary = df_binary.astype(bool).astype(int)

df_cardio_1 = df_binary.drop(columns=['cardio_0'])
df_cardio_0 = df_binary.drop(columns=['cardio_1'])

_, rules_cardio_0 = apriori_algo(df_cardio_0.astype('bool'), 0.01, 0.7)
_, rules_cardio_1 = apriori_algo(df_cardio_1.astype('bool'), 0.01, 0.7)

min_lift = 1
# rules  = rules[(rules['lift'] >= min_lift)]
rules_cardio_0 = rules_cardio_0.sort_values(by=['confidence'], ascending=False)
rules_cardio_1 = rules_cardio_1.sort_values(by=['confidence'], ascending=False)

start_time = time.time()
_, rules = apriori_algo(df_binary, 0.1, 0.7)
end_time = time.time()
elapsed_time_apriori = end_time - start_time

print(f"The runtime of the apriori algorithm is: {elapsed_time_apriori} seconds.")

desired_antecedent_len = 3
filtered_rules_cardio_0 = get_filtered_rules(rules_cardio_0, desired_antecedent_len)
filtered_rules_cardio_1 = get_filtered_rules(rules_cardio_1, desired_antecedent_len)

print('Cardio_0  :', len(filtered_rules_cardio_0))

filtered_rules_cardio_1 = [entry.split(' -> ')[0].strip('{}') for entry in filtered_rules_cardio_1]

count = 0
for rule in filtered_rules_cardio_0:
    print(rule)
    count = count + 1
    if count > 10:
        break

print("*******Cardio_1*********** :", len(filtered_rules_cardio_1))
entry_counts = {}
for rule in filtered_rules_cardio_1:
    entries = rule.split(', ')
    for e in entries:
        if e not in entry_counts:
            entry_counts[e] = 0
        entry_counts[e] += 1

# Prepare the data for plotting
labels = list(entry_counts.keys())
counts = list(entry_counts.values())

# Create the bar plot
plt.bar(labels, counts)
plt.xlabel('Entry')
plt.ylabel('Frequency')
plt.title('Frequency of Entries')
plt.xticks(rotation=90)
plt.show()
