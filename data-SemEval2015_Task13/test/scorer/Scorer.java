import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;


/**
 * @author Andrea Moro (andrea8moro@gmail.com)
 *
 * A simple scorer for the SemEval 2015 task 13 on Multilingual All-Words Sense Disambiguation and 
 * Entity Linking. For file format description please refer to the main README file or to the
 * task web page http://alt.qcri.org/semeval2015/task13/
 *
 */
public class Scorer {

	public static void main(String[] args) throws IOException {
		// Check commoand-line arguments.
		if (args.length < 2 || args.length > 3) exit();
		Set<Integer> documents = new HashSet<Integer>();
		if (args.length > 2) {
			if (args[0].matches("-d[0-9,]+")) {
				for (String d : args[0].substring(2).split(","))
					documents.add(Integer.parseInt(d));
			} else exit();
		}
		// Load gold standard and system annotations.
		File gs = new File(args[args.length==2?0:1]);
		if (!gs.exists()) exit();
		File system = new File(args[args.length==2?1:2]);
		if (!system.exists()) exit();
		// Compute measures.
		Double[] m = score(gs, system, documents);
		System.out.println("P=\t"+String.format("%.1f", m[0]*100)+"%");
		System.out.println("R=\t"+String.format("%.1f", m[1]*100)+"%");
		System.out.println("F1=\t"+String.format("%.1f", m[2]*100)+"%");
	}
	
	private static void exit() {
		System.out.println("Scorer [-d1,...,4] gold-standard_key_file system_key_file\n" +
						   "If the option -d is given then the scorer will evaluate only\n" +
						   "instances from the given list of documents.");
		System.exit(0);
	}
	
	public static Double[] score(File gs, File system, Set<Integer> docs) throws IOException {
		// Read the input files.
		Map<String, Set<String>> gsMap = new HashMap<String, Set<String>>();
		readFile(gs, gsMap, docs);
		Map<String, Set<String>> systemMap = new HashMap<String, Set<String>>();
		readFile(system, systemMap, docs);
		// Count how many good and bad answers the system gives.
		double ok = 0, notok = 0;
		for (String key : systemMap.keySet()) {
			// If the fragment of text annotated by the system is not contained in the gold
			// standard then skip it.
			if (!gsMap.containsKey(key)) continue;
			// Handling multiple answers for a same fragment of text.
			int local_ok = 0, local_notok = 0;
			for (String answer : systemMap.get(key)) {
				if (gsMap.get(key).contains(answer)) local_ok++;
				else local_notok++;
			}
			ok += local_ok/(double)systemMap.get(key).size();
			notok += local_notok/(double)systemMap.get(key).size();
		}
		// Compute precision, recall and f1 scores.
		Double[] m = new Double[3];
		m[0] = ok/(double)(ok+notok);
		m[1] = ok/(double)gsMap.size();
		m[2] = (2*m[0]*m[1]) / (m[0]+m[1]);
		return m;
	}
	
	public static void readFile(File file, Map<String, Set<String>> map, Set<Integer> docs) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(file));
		String l;
		while ((l = in.readLine()) != null) {
			String[] ll = l.split("\t");
			if (ll.length<3) continue;
			// Check if we are interested in this doc.
			int d = Integer.parseInt(ll[0].substring(ll[0].indexOf('d')+1, ll[0].indexOf('.')));
			if (docs != null && docs.size() > 0 && !docs.contains(d)) continue;
			// Update the map with a new set of answers.
			if (!map.containsKey(ll[0]+ll[1])) map.put(ll[0]+ll[1], new HashSet<String>()); 
			for (int i = 2; i < ll.length; i++) map.get(ll[0]+ll[1]).add(ll[i].toLowerCase().replaceAll("_", " "));
		}
		in.close();
	}
}
