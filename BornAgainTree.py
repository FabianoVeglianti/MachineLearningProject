import os
import sys
import pandas as pd
source_path = os.path.abspath('../src')
sys.path.append(source_path)
import datasets as ds
import random_forests as rf
import persistence as tree_io
import visualization as tree_view

class BornAgainTree:

	def __init__(self):
		self._forestMaxDepth
		self._forestNumEstimators
		self._forestCriterion
		self._forestMaxFeatures
		self._numFold
		self._bornAgain = None

	def writeForestOnFile(filename, trees):
		firstTree = trees[0].tree_
		n_nodes = firstTree.node_count
		children_left = firstTree.children_left
		children_right = firstTree.children_right
		feature = firstTree.feature
		threshold = firstTree.threshold

		node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
		is_leaves = np.zeros(shape=n_nodes, dtype=bool)
		stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
		while len(stack) > 0:
		    # `pop` ensures each node is only visited once
		    node_id, depth = stack.pop()
		    node_depth[node_id] = depth

		    # If the left and right child of a node is not the same we have a split
		    # node
		    is_split_node = children_left[node_id] != children_right[node_id]
		    # If a split node, append left and right children and depth to `stack`
		    # so we can loop through them
		    if is_split_node:
			stack.append((children_left[node_id], depth + 1))
			stack.append((children_right[node_id], depth + 1))
		    else:
			is_leaves[node_id] = True
		
		scriptDir= os.path.abspath('')
		parent = os.path.abspath(os.path.join(scriptDir, os.pardir)) #Credo che vada tolto nel nostro codice, ma non ne sono sicuro
		relativePath = "src/resources/forests/Loan/" +filename[0:len(filename)-3] + "RF" + str(n_fold) + ".txt"
		absolutePath = os.path.join(parent, relativePath)
		f = open(absolutePath, 'w')
		f.write("DATASET_NAME: " + filename+"\n")
		f.write("ENSEMBLE: RF\n")
		f.write("NB_TREES: " + str(n_estimators)+"\n")
		f.write("NB_FEATURES: " + str(features.size)+"\n")
		f.write("NB_CLASSES: 2\n")
		f.write("MAX_TREE_DEPTH: " + str(max_depth)+"\n")
		f.write("Format: node / node type (LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n")

		for j in range(0, len(trees)):
		    f.write("\n[TREE " + str(j)+"]\n")
		    tree = trees[j].tree_
		    n_nodes = tree.node_count
		    f.write("NB_NODES: " + str(n_nodes)+"\n")

		    children_left = tree.children_left
		    children_right = tree.children_right
		    feature = tree.feature
		    threshold = tree.threshold
		    value = tree.value

		    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
		    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
		    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
		    while len(stack) > 0:
			# `pop` ensures each node is only visited once
			node_id, depth = stack.pop()
			node_depth[node_id] = depth

			# If the left and right child of a node is not the same we have a split
			# node
			is_split_node = children_left[node_id] != children_right[node_id]
			# If a split node, append left and right children and depth to `stack`
			# so we can loop through them
			if is_split_node:
			    stack.append((children_left[node_id], depth + 1))
			    stack.append((children_right[node_id], depth + 1))
			else:
			    is_leaves[node_id] = True

		    for i in range(n_nodes):
			f.write(str(i) + " ")
			if is_leaves[i]:
			    f.write("LN ")
			    f.write(str(children_left[i])+ " " + str(children_right[i]) + " " + str(feature[i]) + " " + str(threshold[i]) + " " + str(node_depth[i]) + " ")
			    if(value[i][0][0]>=value[i][0][1]):
				f.write("0")
			    else:
				f.write("1")
			    
			else:
			    f.write("IN ")
			    f.write(str(children_left[i])+ " " + str(children_right[i]) + " " + str(feature[i]) + " " + str(threshold[i]) + " " + str(node_depth[i]) + " ")
			    f.write("-1")
			f.write("\n")
		f.write("\n")
		f.flush()
		f.close()

	def prepareData(data_x, data_y, isTrain, numFold):
		scriptDir= os.path.abspath('')
		parent = os.path.abspath(os.path.join(scriptDir, os.pardir)) #Credo che vada tolto nel nostro codice, ma non ne sono sicuro
		if isTrain:
			relativePath = "src/resources/datasets/Loan/" +filename[0:len(filename)-3] + "train" + str(numFold) + ".csv"
		else:
			relativePath = "src/resources/datasets/Loan/" +filename[0:len(filename)-3] + "test" + str(numFold) + ".csv"		
		absolutePath = os.path.join(parent, relativePath)
		data = pd.concat([data_x, data_y], axis=1)
		data.to_csv(absolutePath, index=False)

	def CreateRandomForestByCV(filename, train_x, train_y, forestMaxDepth, forestNumEstimators,  forestCriterion, forestMaxFeatures, numFold, metric):
		self.prepareData(train_x, train_y, True, numFold)

		
		self._forestMaxDepth = forestMaxDepth
		self._forestNumEstimators = forestNumEstimators
		self._forestCriterion = forestCriterion
		self._forestMaxFeatures = forestMaxFreatures
		self._numFold = numFold
		myclassifier = RandomForestClassifier(random_state=42, criterion = forestCriterion, n_estimators = forestNumEstimators, max_features = forestMaxFreatures, max_depth =forestMaxDepth)

		params = {}

		gscv_rf = GridSearchCV(myclassifier, param_grid = params, scoring=metric, cv=numFold, refit=True)

		gscv_rf.fit(train_x, train_y.loan_status)
		
		rf = gscv_rf.best_estimator_
		trees = rf.estimators_

		self.writeForestOnFile(filename, trees)
		
	
	def getBornAgainTree(test_x, test_y):
		self.prepareData(test_x, test_y, False, self._numFold)


		current_dataset = "Loan"
		current_fold = self._numFold
		n_trees = self.forestNumEstimators
		current_obj = 4
		using_cplex = False

		df_train, df_test, ds_infos = ds.load(current_dataset, current_fold)
		X_train, y_train = df_train.iloc[:,:-1].values, df_train.iloc[:,-1].values
		X_test, y_test = df_test.iloc[:,:-1].values, df_test.iloc[:,-1].values


		random_forest, random_forest_file = rf.load(X_train, y_train, current_dataset,
                                            current_fold, n_trees, return_file=True)
		rf_trees = [e.tree_ for e in random_forest.estimators_]

		if 0 == os.system('make --directory=../src/born_again_dp {} > buildlog.txt'.format('withCPLEX=1' if using_cplex else '')):
		    print('Dynamic Program was successful built.')
		else:
		    print('Error while compiling the program with the make commend. Please verify that a suitable compiler is available.')
		    os.system('make --directory=../src/born_again_dp')

		
		# Calling executable to compute Born-Again Tree
		born_again_file = "{}.BA{}".format(current_dataset, current_fold)
		ret = subprocess.run(['../src/born_again_dp/bornAgain',
				random_forest_file,
				born_again_file,
				'-trees', str(n_trees),
				'-obj', str(current_obj)], stdout=subprocess.PIPE)

		print("Executed command: \"{}\"\n".format(' '.join(ret.args)))
		print(ret.stdout.decode('utf-8'))
		if ret.returncode != 0:
		    print(ret.stderr.decode('utf-8'))
		print('Program exited with code {}.'.format(ret.returncode))

		# Visualizing...
		born_again = tree_io.classifier_from_file(born_again_file+".tree", X_train, y_train, pruning=False)
		#born_again_pruned = tree_io.classifier_from_file(born_again_file+".tree", X_train, y_train, pruning=True)
		self._bornAgain = born_again
		return born_again

	def predict(x):
		yPred = self._bornAgain.predict(x)		
		return yPred


		
