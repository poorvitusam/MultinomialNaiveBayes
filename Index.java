import java.util.ArrayList;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map.Entry;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.*;

/**
 * @author poorvitusam 
 * This is the class for multinomial Naive Bayes
 * Classification model using laplace 1 smoothing
 *
*/
public class Index {
	/* Directory to maintain words */
	public static LinkedHashMap<String, Integer> vocabDir = new LinkedHashMap<String, Integer>();
	public static LinkedHashMap<String, Integer> hamVocabDir = new LinkedHashMap<String, Integer>();
	public static LinkedHashMap<String, Integer> spamVocabDir = new LinkedHashMap<String, Integer>();

	/* Directory for test data */
	public static LinkedHashMap<String, Integer> testHamVocabDir = new LinkedHashMap<String, Integer>();
	public static LinkedHashMap<String, Integer> testSpamVocabDir = new LinkedHashMap<String, Integer>();

	/* List to maintain all conditional probablity */
	public static LinkedHashMap<String, Double> hamConditionalProb = new LinkedHashMap<String, Double>();
	public static LinkedHashMap<String, Double> spamConditionalProb = new LinkedHashMap<String, Double>();

	/* List for storing the documents segregated as per the folder */
	public static ArrayList<File> allDocs = new ArrayList<File>();
	public static ArrayList<File> hamDocs = new ArrayList<File>();
	public static ArrayList<File> spamDocs = new ArrayList<File>();

	/* Doc list for test data */
	public static ArrayList<File> test_hamDocs = new ArrayList<File>();
	public static ArrayList<File> test_spamDocs = new ArrayList<File>();

	/* Doc counts */
	public static int n_ham_count = 0;
	public static int n_spam_count = 0;
	public static int n_count = 0;
	public static int n_test_count = 0;
	
	/* Priors */
	public static double prior_ham = 0;
	public static double prior_spam = 0;

	public static int word_count_ham = 0;
	public static int word_count_spam = 0;
	public static int distinct_word_count = 0;
	
	public static String TEXTFILE=".txt";
	public static String HAM="ham";
	public static String SPAM="spam";

	/* Initialise accuracy */
	public static int accurracy = 0;

	public static void main(String args[]) {
		try {
			String mainDir = args[0];
			String testDir = args[1];
			
			readTrainingData(mainDir);
			trainmMultinomialNB();
			checkWithTest(testDir);
			System.out.
				println((double) accurracy * 100 / ((double) test_hamDocs.size() + 
						(double) test_spamDocs.size()));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

/*
 * Reading the Training data
 */
	private static void readTrainingData(String mainDir) {
		n_count = listf(mainDir, allDocs);
		extractAllVocab(allDocs, vocabDir, true);

		/* Run for all sub folders for train data */
		File directory = new File(mainDir);
		File[] fList = directory.listFiles();
		for (File file : fList) {
			if (file.isDirectory()) {
				if (file.getName().equals("ham")) {
					n_ham_count = listf(file.getPath(), hamDocs);
					word_count_ham = extractAllVocab(hamDocs, hamVocabDir, false);
				} else {
					n_spam_count = listf(file.getPath(), spamDocs);
					word_count_spam = extractAllVocab(spamDocs, spamVocabDir, false);
				}
			}
		}
	}

	private static void trainmMultinomialNB() {
		prior_ham = (double) n_ham_count / (double) n_count;
		calculateConditionalProbablity(vocabDir, hamConditionalProb, hamVocabDir, word_count_ham);
		
		prior_spam = (double) n_spam_count / (double) n_count;
		calculateConditionalProbablity(vocabDir, spamConditionalProb, spamVocabDir, word_count_spam);
	}

	private static void checkWithTest(String testDir) {
		File test_directory = new File(testDir);
		File[] test_fList = test_directory.listFiles();
		for (File file : test_fList) {
			if (file.isDirectory()) {
				if (file.getName().equals(HAM)) {
					listf(file.getPath(), test_hamDocs);
					
					for (int i = 0; i < test_hamDocs.size(); i++) {
						try {
							FileReader test_file = new FileReader(test_hamDocs.get(i));
							BufferedReader br = new BufferedReader(test_file);
							getData(br, testHamVocabDir, false);
							calculateAccuracy(testHamVocabDir, HAM);
						} catch (Exception e) {
							e.printStackTrace();
						}
						testHamVocabDir.clear();
					}

				} else {
					listf(file.getPath(), test_spamDocs);
					for (int i = 0; i < test_spamDocs.size(); i++) {
						try {
							FileReader test_file = new FileReader(test_spamDocs.get(i));
							BufferedReader br = new BufferedReader(test_file);
							getData(br, testSpamVocabDir, false);
							calculateAccuracy(testSpamVocabDir, SPAM);
						} catch (Exception e) {
							e.printStackTrace();
						}
						testSpamVocabDir.clear();
					}
				}

			}
		} /*end runtest */
	}

	/* 
	 * Helper function to extract all files from a directory returning the file count 
	 * 
	 */
	private static int listf(String directoryName, ArrayList<File> files) {
		File directory = new File(directoryName);
		int n = 0;
		
		File[] fList = directory.listFiles();
		for (File file : fList) {
			if (file.getName().contains(TEXTFILE)) {
				n++;
				files.add(file);
			} else if (file.isDirectory()) {
				n += listf(file.getAbsolutePath(), files);
			}
		}
		return n;
	}

	/**
	 * Scans each line from all files in the given folder passed in parameter dir
	 * @param docs:Documents which are scanned
	 * @param dir: Store data extracted from documents in this hashmap
	 * @param calculateDistinct: Count for distinct words
	 * @return
	 */
	private static int extractAllVocab(ArrayList<File> docs, LinkedHashMap<String, Integer> dir,
			boolean calculateDistinct) {
		int count = 0;
		for (int i = 0; i < docs.size(); i++) {
			try {
				FileReader file = new FileReader(docs.get(i));
				BufferedReader br = new BufferedReader(file);
				count += getData(br, dir, calculateDistinct);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return count;
	}

	
	/**
	 * Extracts each line passed by extractAllVocab for words and stores them in the hashmap
	 * @param br:Line passed from extractAllVocab
	 * @param dir:Store in this hashmap
	 * @param calculateDistinct: Count for distinct words
	 * @return
	 */
	private static int getData(BufferedReader br, LinkedHashMap<String, Integer> dir, boolean calculateDistinct) {
		String line;
		int count = 0;
		String regex = "[\\w']+";

		try {
			while ((line = br.readLine()) != null) {
				Pattern pattern = Pattern.compile(regex);
				Matcher matcher = pattern.matcher(line);
				while (matcher.find()) {
						count++;
						
						if (dir.containsKey(matcher.group())) {
							for (Entry<String, Integer> entry : dir.entrySet()) {
								if (entry.getKey().equals(matcher.group())) {
									entry.setValue(entry.getValue() + 1); /* If word is found increment */
									break;
								}
							}
						} else {
							
							dir.put(matcher.group(), 1); /* If word not present then add the word in the directory */
							if (calculateDistinct) {
								distinct_word_count++;
							}
						}
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
		return count;
	}

	private static void calculateConditionalProbablity(LinkedHashMap<String, Integer> main,
			LinkedHashMap<String, Double> conditionalProb, LinkedHashMap<String, Integer> dir, int word_count) {
		for (Entry<String, Integer> entry : main.entrySet()) {
			int val = 0;
			if (dir.containsKey(entry.getKey())) {
				for (Entry<String, Integer> e : dir.entrySet()) {
					if (e.getKey().equals(entry.getKey())) {
						val = e.getValue();
					}
				}
			}
			double prob = ((double) val + 1) / ((double) distinct_word_count + (double) word_count);
			conditionalProb.put(entry.getKey(), prob);
		}
	}

	private static void calculateAccuracy(LinkedHashMap<String, Integer> testVocabDir, String doc_class) {
		double hamLikelihood = Math.log(prior_ham);
		double spamLikelihood = Math.log(prior_spam);

		for (Entry<String, Integer> entry : testVocabDir.entrySet()) {
			hamLikelihood = hamLikelihood + getLikelihoodTerm(hamConditionalProb, entry.getKey(), entry.getValue());
			spamLikelihood = spamLikelihood + getLikelihoodTerm(spamConditionalProb, entry.getKey(), entry.getValue());
		}
		String predict_class = "";
		if (hamLikelihood > spamLikelihood) {
			predict_class = HAM;
		} else {
			predict_class = SPAM;
		}
		if (predict_class == doc_class) {
			accurracy++;
		}

	}

	private static double getLikelihoodTerm(LinkedHashMap<String, Double> dir, String word_name, int word_count) {
		double score = 0;
		for (Entry<String, Double> entry : dir.entrySet()) {
			if (entry.getKey().equals(word_name)) {
				for (int i = 0; i < word_count; i++) {
					score += Math.log(entry.getValue());
				}
				break;
			}
		}
		return score;
	}

}
