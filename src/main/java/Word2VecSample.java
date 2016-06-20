import org.apache.lucene.analysis.util.TokenizerFactory;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Created by cdathuraliya on 8/27/15.
 */
public class Word2VecSample {
    public static void main(String[] args) {
        String filePath = null;
        try {
            filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println();
        System.out.println("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = null;
        try {
            iter = UimaSentenceIterator.createWithPath(filePath);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Split on white spaces in the line to get words
        DefaultTokenizerFactory t = new DefaultTokenizerFactory();
        //t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache();
        WeightLookupTable table = new InMemoryLookupTable.Builder()
                .vectorLength(100)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build();

        System.out.println("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5).iterations(1)
                .layerSize(100).lookupTable(table)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache).seed(42)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        System.out.println("Fitting Word2Vec model....");
        try {
            vec.fit();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Writing word vectors to text file....");
        // Write word
        try {
            WordVectorSerializer.writeWordVectors(vec, "output.txt");
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Closest 10 words '#politics':");
        Collection<String> lst = vec.wordsNearest("#politics", 10);
        System.out.println(lst);

        System.out.println("Evaluate model....");
        double sim1 = vec.similarity("#GenElecSL", "#politics");
        double sim2 = vec.similarity("#JVPSL", "#politics");
        double sim3 = vec.similarity("දේශපාලන", "#politics");
        double sim4 = vec.similarity("viciously", "#politics");
        double sim5 = vec.similarity("during", "#politics");

        System.out.println("Similarity between '#GenElecSL' and '#politics': " + sim1);
        System.out.println("Similarity between '#JVPSL' and '#politics': " + sim2);
        System.out.println("Similarity between 'දේශපාලන' and '#politics': " + sim3);
        System.out.println("Similarity between 'viciously' and '#politics': " + sim4);
        System.out.println("Similarity between 'during' and '#politics': " + sim5);
        /*Collection<String> similar = vec.wordsNearest("model", 10);
        System.out.println("Similar words to 'model' : " + similar);*/
    }
}
