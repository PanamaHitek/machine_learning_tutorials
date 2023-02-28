package com.panamahitek.sklearn;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class MammographicMassClassifier {

    public static void main(String[] args) {
        // Load the test dataset from CSV file
        String filename = "../../datasets/mammographic_mass/test.csv";
        double[][] test_data = loadCsv(filename);
        int[] expected_results = new int[test_data.length];
        for (int i = 0; i < test_data.length; i++) {
            expected_results[i] = (int) test_data[i][0];
        }

        // Define the threshold for classifying examples as positive
        double threshold = 0.0;

        // Make a prediction for each sample in the test dataset
        int valid_results = 0;
        long start_time = System.currentTimeMillis();
        for (int i = 0; i < test_data.length; i++) {
            double[] input = new double[5];
            for (int j = 1; j <= input.length; j++) {
                input[j - 1] = test_data[i][j];
            }
            
            double output = Model.score(input);
            int result = (output > threshold) ? 1 : 0;

            // Check if the predicted result matches the expected result
            String outcome = "Fail";
            if (result == expected_results[i]) {
                valid_results++;
                outcome = " OK ";
            }

            // Print the classification results for each sample
            System.out.println("Nº " + (i + 1) + " | Expected result: " + expected_results[i] + " | Obtained result: " + result + " | " + outcome + " | Accuracy: " + String.format("%.2f", (double) valid_results / (i + 1) * 100) + "%");
        }
        long end_time = System.currentTimeMillis();
        double testing_time = (double) (end_time - start_time) / 1000.0;

        // Print the final testing results
        System.out.println("---------------------------");
        System.out.println("Results");
        System.out.println("---------------------------");
        System.out.println("Testing samples: " + test_data.length);
        System.out.println("Testing time: " + testing_time + " s");
        System.out.println("Testing accuracy: " + String.format("%.2f", (double) valid_results / test_data.length * 100) + "%");
    }

    public static double[][] loadCsv(String filename) {
        // Load a CSV file into a 2D double array
        try {
            Scanner scanner = new Scanner(new File(filename));
            scanner.useDelimiter(",");
            int rows = 0;
            int cols = 0;
            while (scanner.hasNextLine()) {
                rows++;
                String[] line = scanner.nextLine().split(",");
                cols = line.length;
            }
            scanner.close();
            double[][] data = new double[rows][cols];
            scanner = new Scanner(new File(filename));
            scanner.useDelimiter(",");
            int row = 0;
            while (scanner.hasNextLine()) {
                String[] line = scanner.nextLine().split(",");
                for (int col = 0; col < cols; col++) {
                    data[row][col] = Double.parseDouble(line[col]);
                }
                row++;
            }
            scanner.close();
            return data;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }
}
