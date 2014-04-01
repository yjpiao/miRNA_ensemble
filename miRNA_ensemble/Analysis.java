package miRNA_ensemble;

import java.util.Random;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Filename : ECBGS.java
 * Purpose  : This class is used to compute the required mathematical computations for the 
 * ECBGS (Ensemble Correlation-Based Gene Selection) algorithm.
 * @author  : Yongjun Piao
 *
 */

public class Analysis 
{
	private Instances instances;
	private Instances filteredInstances;
	private double[][] data;

	private Vector suList = null;
	private Vector suListDup = null;
	private Vector relevantSubset = null;
	private Vector finalSubset = null;
	
	private double threshold;   /* threshold for selecting relevant genes */
	private int numberOfClassifiers; /* the nubmer of classifiers in ensemble */
	private String classifierType; /*Classifier type: C for C4.5, S for SVM */
			
			
	public Analysis(Instances instances, Instances filterdInstances, double[][] data, String threshold, String cType, String number)
	{
		this.instances = instances;
		this.filteredInstances = filterdInstances;
		this.data = data;
		this.threshold = Double.parseDouble(threshold);
		this.classifierType = cType;
		this.numberOfClassifiers = Integer.parseInt(number);
	}
	
	/** This function returns the finally selected subset by ECBGS.
	 *  @param
	 * 	@return Returns the index of the selected features.
	 */
	Vector getFinalSubset()
	{
		return finalSubset;
	}
	
	/** The analysis function applies the ECBGS algorithm and outputs the selected feature subset. 
	 * @throws Exception */
	void analysis() throws Exception
	{
		Classifier[] classifiers = new FilteredClassifier[numberOfClassifiers];

		/*
		 * STEP 1: relevance analysis
		 */
		int numAttr = filteredInstances.numAttributes()-1;
		int len = 0;
		
		suList = new Vector();
		suListDup = new Vector();
		relevantSubset = new Vector();
		
		/* Initializing 'suList' */
		for(int i = 0; i < numAttr; i++)
		{
			/* Calculating Symmetrical Uncertainty with respect to the class */
			double su = SU(i, numAttr);
			suList.add(su);
			suListDup.add(su);
			if((Double)suList.elementAt(i) > threshold)
				len++;
		}
		
		int[] removing = new int[numAttr];
		
		for(int k = 0; k < numAttr; k++) {
			removing[k] = -1;
		}
		
		double max;
		int maxIndex;
		
		for(int i=0; i<len; i++)
		{
			max = 0;
			maxIndex = -1;
		 	
		 	for(int j=0;j<numAttr;j++)
		 	{
			 	if((Double)suListDup.elementAt(j) >= max)
			 	{
			 		max=(Double) suListDup.elementAt(j);
			 		maxIndex=j;
			 	}
		 	}
		 	
		 	relevantSubset.add(maxIndex);
		 	suListDup.set(maxIndex, (double)0);/* Removing the max element in order to get the next 
		 										maximum element in the next iteration */
		}
		
		/** Print the attribute index of relevant subset. This is used for debugging process.
		 *  for(int i = 0; i < len; i++)
		 *		System.out.println(relevantSubset.elementAt(i));
		*/
		
		/* STEP 2: redundancy analysis 
		 * Generates a number of feature subsets, and each subset is evaluated by SVM.
		*/
		
		Vector temp = new Vector();
		Vector removedSubset = new Vector();
		finalSubset = new Vector();
		int startPoint=0, fp, fq, fqi, iteration = 0;
		double accuracy = 0.0, maxAccuracy = 0.0;
		int classiferCount = 0;
		
		System.out.println("Processing.... Please wait.....");
		
		while(true)
		{
			temp = (Vector) relevantSubset.clone();
			double maxRemoved = -1;
			
			/* Set the starting point */
			if(removedSubset.isEmpty())
				startPoint = (Integer) relevantSubset.firstElement();
			else{
				for(int i = 0; i < removedSubset.size(); i++){
					if( (Double)suList.elementAt((Integer) removedSubset.elementAt(i)) > maxRemoved)
					{
						maxRemoved = (Double)suList.elementAt((Integer) removedSubset.elementAt(i));
						startPoint = (Integer)removedSubset.elementAt(i);
					}
				}
			}
			
			int numRemove = temp.indexOf(startPoint);
			
			/* Remove the features that have rankings higher than the starting point */
			if(numRemove != 0)
				for(int i = 0; i < numRemove; i++)
					temp.removeElementAt(0);
			
			fp = startPoint;
			removedSubset.clear();
			
			while(true)
			{
				if(fp == (Integer)temp.lastElement())
					break;
				
				fq = (Integer)temp.elementAt(temp.indexOf(fp)+1);
				
				while(true)
				{
					fqi = temp.indexOf(fq);
					
					if(SU(fp, fq) >= (Double)suList.elementAt(fq))
					{
						removedSubset.add(fq);
						if(fq == (Integer)temp.lastElement())
						{
							temp.removeElement(fq);
							break;
						}
						else
						{
							temp.removeElement(fq);
							fq = (Integer) temp.elementAt(fqi);
						}
					}
					else
					{
						if(fq == (Integer)temp.lastElement())
							break;
						fq = (Integer)temp.elementAt(fqi+1);
					}
						
				}
				
				if(fp == (Integer)temp.lastElement())
					break;
				
				fp = (Integer)temp.elementAt(temp.indexOf(fp)+1);
			}
			
			for(int k = 0; k < numAttr; k++) {
				for(int m = 0; m < temp.size(); m++) {
					if((k) == (Integer)temp.elementAt(m))
						removing[k] = 0;
				}
			}
			
			/* Constructing Ensemble */
			
			int n = 0;
			int count = 0;
			
			for(int m = 0; m < numAttr; m++) {
				if(removing[m] == -1)
					count++;
			}
			
			int[] deleteIndex = new int[count];
			
			for(int m = 0; m < numAttr; m++) {
				if(removing[m] == -1) {
					deleteIndex[n] = m;
					n++;
				}
			}

			Remove remove = new Remove();
			remove.setAttributeIndicesArray(deleteIndex);
		
			J48 j48 = new J48();
			SMO smo = new SMO();
			FilteredClassifier fc = new FilteredClassifier();
			
			if(classifierType.equals("C"))
				fc.setClassifier(j48);
			else
				fc.setClassifier(smo);
			
			fc.setFilter(remove);

			classifiers[classiferCount] = fc;
			classiferCount++;
			
			if(removedSubset.isEmpty() || iteration == numberOfClassifiers-1)
				break;
			
			iteration++;
		}
		
		Vote vote = new Vote();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "AVG";
		vote.setOptions(options);
		vote.setClassifiers(classifiers);
		
		double[] accuracies = new double[1];
		
		for(int i = 0; i < 1; i++)
		{
			Evaluation eval = new Evaluation(instances);
			eval.crossValidateModel(vote, instances, 10, new Random(i));
			
			double erroRate = eval.errorRate()*100;
			erroRate  = Math.round(erroRate * 1000);
			erroRate = erroRate / 1000;
			
			double s = eval.weightedTruePositiveRate()*100;
			s  = Math.round(s * 1000);
			s = s / 1000;
			
			double p = eval.weightedTrueNegativeRate()*100;
			p  = Math.round(p * 1000);
			p = p / 1000;
			
			double a = eval.weightedAreaUnderROC()*100;
			a  = Math.round(a * 1000);
			a = a / 1000;
			
			accuracies[i] = eval.correct()/instances.numInstances();
		}
			
		for(double acc : accuracies)
			System.out.print(acc+", ");
	}
	
	/** This function returns the classification accuracy of a feature subset.
	 *  @param a subset of features
	 * 	@return Returns the classification accuracy.
	 */
	
	/** This function returns the entropy of an attribute 
	 *  pointed by 'index' (ranges from 0 to numAttrs+1)
	 *  @param	index Index of the attribute
	 * 	@return Returns the entropy of the feature/attribute.
	 */
	double entropy(int attrIndex){
		
		double entropy = 0;
		double temp;
		
		for(int i = 0; i < filteredInstances.attributeStats(attrIndex).distinctCount ; i++)
		{
			temp = partialProb(attrIndex, i);
			if(temp != (double)0)
				entropy += temp *(Math.log(temp)/Math.log((double)2.0));
		}
		
		return -entropy;
	}
	
	/** This function computes the conditional entropy of the attribute One (mentioned by indexOne),
	 *  given the attribute Two (mentioned by indexTwo)
	 *  @param	indexOne Index of attribute One
	 *  @param	indexOne Index of attribute One
	 * 	@return	Partial Probability
	 */
	double partialProb(int attrIndex, int attrValue){
		
		int count = 0;
		
		for(int i = 0; i < instances.numInstances(); i++)
			if(data[i][attrIndex] == attrValue)
				count++;
		
		if(count != 0)
			return (double)count/(double)instances.numInstances();
		else
			return (double)0;
	}
	
	/** This function computes the conditional entropy of the attribute One (mentioned by indexOne),
	 *  given the attribute Two (mentioned by indexTwo)
	 *  @param	indexOne Index of attribute One
	 *  @param	indexOne Index of attribute One
	 * 	@return	Conditional Probability of One given Two
	 */	
	double condEntropy(int indexOne,int indexTwo)
	{
		double ans=0,temp,temp_ans,cond_temp;
		
		
		for(int j=0; j < filteredInstances.attributeStats(indexTwo).distinctCount; j++)
		{
			temp=partialProb(indexTwo,j);
			temp_ans=0;
			
			for(int i=0;i < filteredInstances.attributeStats(indexOne).distinctCount; i++)
			{
				cond_temp=partialCondProb(indexOne,i,indexTwo,j);
				if(cond_temp != (double)0)
					temp_ans += cond_temp *(Math.log(cond_temp)/Math.log((double)2.0));
			}
			ans+=temp*temp_ans;
		}
		return -ans;
	}
	
	/** This function computes the probability of feature/attribute One(given by indexOne) taking
	 *  value 'valueOne', given feature Two(given by indexTwo) taking value 'valueTwo'
	 *  @param indexOne Index of feature One
	 *  @param valueOne Value of feature One
	 *  @param indexTwo Index of feature Two
	 *  @param valueTwo Value of feature Two
	 * 	@return
	 */	
	double partialCondProb(int indexOne,int valueOne,int indexTwo,int valueTwo)
	{
		int num=0,den=0;
		
		for(int i=0; i < instances.numInstances(); i++)
		{	
			if(data[i][indexTwo] == valueTwo)
			{
				den++;
				if(data[i][indexOne] == valueOne)
					num++;
			}
		}
		
		if(den!=0)
			return (double)num/(double)den;
		else
			return (double)0;
	}
	
	/** This function computes the information gain of feature 'indexOne' given feature 'indexTwo'
	 *  IG(indexOne,indexTwo) => ENTROPY(indexOne) - condEntropy(indexOne,indexTwo)
	 *  @param indexOne	feature One
	 *  @param indexTwo	feature Two
	 *  @return Returns the Information Gain
	 */	
	double informationGain(int indexOne,int indexTwo)
	{
		return entropy(indexOne) - condEntropy(indexOne,indexTwo);
	}
	
	/** This function computes the Symmetrical Uncertainity of the features pointed by 'indexOne'
	 *  and 'indexTwo'
	 * 
	 *  @param indexOne Feature One
	 *  @param indexTwo Feature Two
	 *  @return Returns Symmetrical Uncertainty
	 */
	double SU(int indexOne,int indexTwo)
	{
		double ig,e1,e2;
		
		ig=informationGain(indexOne,indexTwo);
		e1=entropy(indexOne);
		e2=entropy(indexTwo);
		
		if((e1+e2) !=(double)0)
			return((double)2 * (ig/(e1+e2)));
		else
		return (double)1;
	}
}
