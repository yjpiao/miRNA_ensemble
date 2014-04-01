package miRNA_ensemble;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 * Filename : DataHolder.java
 * Purpose  : This class is used to store the data.
 * 			  Initially, the data is stored as a Instances data structure, which was implemented in Weka.
 * 			  Later, the data is converted into a two dimensional array of primitive types, for faster manipulation
 * 
 * @author  : Yongjun Piao
 *
 */

public class Data {
	
	private String fileName = null;
	private Instances instances = null;				
	private Instances filteredInstances = null;			/* Store the discretized data */
	private double[][] data = null;						/* 2D representation of the data */	
	
	public Data(String fileName){
		this.fileName = fileName;
		readData();
		discretization();	/* Discretize the data */
		convertData();
	}
	
	/** Method: getData()
	 *  @param	none
	 * 	@return 2D array of the instances
	 */
	public double[][] getData() {
		return data;
	}
	
	/** Method: getInstances()
	 *  @param 	none
	 * 	@return instances
	 */
	public Instances getInstances() {
		return instances;
	}

	/** Method: getFilteredInstances()
	 *  @param 	none
	 * 	@return dicretized instances
	 */
	public Instances getFilteredInstances(){
		return filteredInstances;
	}
	
	
	/** Method: readData()
	 *  Description: Read the data from file. Note: The data file should be an ".arff" format.
	 *  @param	none
	 * 	@return none
	 */
	void readData(){
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName + ".arff"));
			instances = new Instances(reader);
			reader.close();
			instances.setClassIndex(instances.numAttributes()-1);
		} catch (FileNotFoundException e) {
			System.out.println("THE FILE: " + fileName + " WAS NOT FOUND.");
			System.exit(0);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	/** Method: convertData()
	 *  Description: Convert the data into a 2D array.
	 *  @param	none
	 * 	@return none
	 */
	void convertData(){
		data = new double[filteredInstances.numInstances()][filteredInstances.numAttributes()];
		
		for(int i = 0; i < filteredInstances.numInstances(); i++){
			for(int j = 0; j < filteredInstances.numAttributes(); j++)
				data[i][j] = (double)filteredInstances.instance(i).value(j);
		}
	}
	
	/* Print the 2D array data. This method is used for debuging process */
	void printConvertedData(){
		for(int i = 0; i < data.length; i++){
			for(int j = 0; j < data[i].length; j++)
				System.out.print(data[i][j] + "\t\t\t");
			System.out.println();
		}
	}
	
	/** Method: discretization()
	 *  Description: Discretize the data using weka's Discretization class.
	 *  @param	none
	 * 	@return none
	 */
	void discretization(){
		Discretize discretize = new Discretize();
		try {
			discretize.setInputFormat(instances);
			discretize.setUseBetterEncoding(true);
			filteredInstances = Filter.useFilter(instances, discretize);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
