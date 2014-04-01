
package miRNA_ensemble;

import weka.core.Instances;

/**
 * Filename : TestClass.java
 * Purpose  : This class is the entry point, which illustrates the usage of the ECBGS algorithm.
 * @author Yongjun Piao
 *
 */

public class TestClass {
	
	public static void main(String[] args) throws Exception {
		
		if( args.length == 4 )
		{	
			//check input parameters
			try {  		  
				if (Float.parseFloat(args[1]) < 0 || Float.parseFloat(args[1]) >1) {
					System.out.println("The relevant threshold must be in range from 0 to 1.");
					System.exit(0);
				}
			}catch(NumberFormatException e) { 
		        System.out.println("The relevant threshold must be a float number\n");
		        System.exit(0);
			}
			
			try {  		  
				if (Integer.parseInt(args[3]) < 1) {
					System.out.println("The number of classifiers must be larger than 1.");
					System.exit(0);
				}
			}catch(NumberFormatException e) { 
		        System.out.println("The number of classifiers must be an integer\n");
		        System.exit(0);
			}
			
			if(args[2].equals("C") || args[2].equals("S")) {
				Data dataHolder = new Data(args[0]);
				
				/* Read data from the file */
				Instances instances = dataHolder.getInstances();
				
				/* Get the discretized data*/
				Instances filteredInstances = dataHolder.getFilteredInstances();
				
				/* Get the data with double[][] format */
				double[][] data = dataHolder.getData();
				
				Analysis analysis = new Analysis(instances, filteredInstances, data, args[1], args[2], args[3]);
				analysis.analysis();
				
				System.out.println("Finished....");	
			}
			else{
				System.out.println("The type of classifier must be a 'C' or 'S'");
			}
		}
		
		else
			System.out.println("Usage: FileName threshold classifier number (ex: TestClass fileName 0 S 20)");
	}
}
