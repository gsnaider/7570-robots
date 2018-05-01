package ar.uba.fi.robots;



import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import ar.uba.fi.robots.sentiment.AnalyzeBagOfWords;



public class Classifier {

	public static void main(String[] args) {
		String filename = null;
		String text="";
		
		if(args.length>1 && args[0].equals("-f")){
			filename=args[1];
		}else if(args.length==1){
			text=args[0];
		}else{
			printhelp();		
		}
		if(filename!=null){
			text=loadfile(filename);
		}
		doProcess(text);		
		
	}
	
	private static void doProcess(String text){
		AnalyzeBagOfWords abw =null;
		try {
			abw = new AnalyzeBagOfWords();
		} catch (IOException e) {
			System.out.println("Se produjo el siguiente error: "+e.getMessage());
			e.printStackTrace();
			System.exit(-1);
		}
		double polarity =abw.doAnalysis(text);
		if(polarity>0){
			System.out.println("polarity: positive; value="+polarity);
		}else{
			System.out.println("polarity: negative; value="+polarity);
		}
	}

	private static void printhelp() {
		System.out.println("Se espera como parametro un texto en inglés o bien el parametro -f"+
				           " y el nombre de un archivo de texto, por ejemplo: ");
		System.out.println("java -jar fastSentimentClassifier.jar \"The movie was great!\" ");
		System.out.println("o bien");
		System.out.println("java -jar fastSentimentClassifier.jar -f critica01.txt ");
		
	}

	private static String loadfile(String filenamepath){
		StringBuilder sb=null;
		try (BufferedReader br = new BufferedReader(new FileReader(filenamepath))) {
			sb=new StringBuilder("");
			String line;
			while ((line = br.readLine()) != null) {
				sb.append(line);
				sb.append("\n");
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
		return sb.toString();
	}
}
