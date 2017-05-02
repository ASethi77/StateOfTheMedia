import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.nytlabs.corpus.NYTCorpusDocument;
import com.nytlabs.corpus.NYTCorpusDocumentParser;
import org.apache.commons.io.FileUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;

public class FuckWithNYT {
    public static void main(String[] args) {
        String nytCorpusRoot = "/opt/nlp_shared/data/news_articles/nytimes/LDC2008T19/data/2007/";
        String outputFilenameBase = "/opt/nlp_shared/data/news_articles/nytimes/nytimes_json/";
        File nytCorpusRootDir = new File(nytCorpusRoot);
        String[] extensions = {"xml"};
        Set<String> topicExclusionSet = topicExclusionSet();
        Iterator<File> nytArticleFileItr = FileUtils.iterateFiles(nytCorpusRootDir, extensions, true);
        Gson gson = new Gson();

        while (nytArticleFileItr.hasNext()) {
            File nytArticleFile = nytArticleFileItr.next();
            NYTCorpusDocument articleDoc = nytFileToDoc(nytArticleFile);

            if (excludeArticle(articleDoc, topicExclusionSet)) continue;

            String articleJson = gson.toJson(articleDoc);

            String articleID = articleDoc.getPublicationYear() + "-" + articleDoc.getPublicationMonth() + "-" + articleDoc.getPublicationDayOfMonth() + "_" + articleDoc.getGuid();
            String outputFilename = outputFilenameBase + articleID + ".json";
//            System.out.println(outputFilename);
            writeJsonToFile(outputFilename, articleJson);

        }
    }

    public static NYTCorpusDocument nytFilePathToDoc(String filepath) {
        File nytArticleFile = new File(filepath);
        NYTCorpusDocumentParser corpusParser = new NYTCorpusDocumentParser();
        return corpusParser.parseNYTCorpusDocumentFromFile(nytArticleFile, false);
    }

    public static NYTCorpusDocument nytFileToDoc(File file) {
        File nytArticleFile = file;
        NYTCorpusDocumentParser corpusParser = new NYTCorpusDocumentParser();
        return corpusParser.parseNYTCorpusDocumentFromFile(nytArticleFile, false);
    }

    public static Set<String> topicExclusionSet() {
        Set<String> s = new HashSet<>();
        s.add("Arts");
        s.add("Automobiles");
        s.add("Books");
        s.add("Corrections");
        s.add("Dining and Wine");
        s.add("Editors' Notes");
        s.add("Health");
        s.add("Home and Garden");
        s.add("Magazine");
        s.add("Movies");
        s.add("New York and Region");
        s.add("Obituaries");
        s.add("Paid Death Notices");
        s.add("Real Estate");
        s.add("Sports");
        s.add("Style");
        s.add("Theater");
        s.add("Travel");
        s.add("Week in Review");
        return s;
    }

    public static void writeJsonToFile(String filename, String json) {
        FileWriter fw = null;
        BufferedWriter bw = null;
        try {
            File f = new File(filename);
            f.createNewFile();

            fw = new FileWriter(filename);
            bw = new BufferedWriter(fw);
            bw.write(json);

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null)
                    bw.close();

                if (fw != null)
                    fw.close();

            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    public static boolean excludeArticle(NYTCorpusDocument articleDoc, Set<String> topicExclusionSet) {
        String onlineSection = articleDoc.getOnlineSection();
        if (onlineSection == null) return false;
        for (String section : onlineSection.split(";")) {
            if (topicExclusionSet.contains(section.trim()))
                return true;
        }
        return false;
    }
}
