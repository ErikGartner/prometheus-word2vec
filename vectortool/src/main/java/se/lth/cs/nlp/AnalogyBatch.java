package se.lth.cs.nlp;

import com.google.common.io.Files;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.jtwig.JtwigModel;
import org.jtwig.JtwigTemplate;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOError;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by marcusk on 2017-03-14.
 */
public class AnalogyBatch {
    private static class Word implements Comparable<Word> {
        final String model;
        final String word;
        final List<Word2vec.Result> ranks;

        public Word(String model, String word, List<Word2vec.Result> ranks) {
            this.model = model;
            this.word = word;
            this.ranks = ranks;
        }

        public String getModel() {
            return model;
        }

        public String getWord() {
            return word;
        }

        public List<Word2vec.Result> getRanks() {
            return ranks;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Word word1 = (Word) o;

            if (!model.equals(word1.model)) return false;
            return word.equals(word1.word);
        }

        @Override
        public int hashCode() {
            int result = model.hashCode();
            result = 31 * result + word.hashCode();
            return result;
        }

        @Override
        public int compareTo(Word o) {
            int wordc = word.compareTo(o.word);
            if(wordc != 0)
                return wordc;

            return model.compareTo(o.model);
        }
    }

    public static void main(String[] args) throws  Exception {
        analogy(Arrays.asList(args));
    }

    public static void analogy(List<String> args) throws Exception {
        //First arg: [words.txt]
        //Second arg: [report.html]
        //Rest of args: models to evaluate:

        int top = 50;

        List<String> words = Files.readLines(new File(args.get(0)), StandardCharsets.UTF_8);
        Collections.sort(words);
        File targetReport = new File(args.get(1));

        List<String> models = args.subList(2, args.size());
        if(!models.stream().allMatch(f -> new File(f + ".vocab").exists() || new File(f + ".vecs").exists())) {
            models.stream().filter(f -> (!new File(f + ".vocab").exists()) || (!new File(f + ".vecs").exists())).forEach(f -> {
                System.out.println("Could not find model " + f);
                throw new RuntimeException("Could not find all models!");
            });
        }

        Map<String,List<Word>> modelWords = models.stream().flatMap(inp -> {
            try {
                System.out.println("Processing " + inp);
                Word2vec w2v = new Word2vec(new File(inp + ".vocab"), new File(inp + ".vecs"));
                ArrayList<Word> output = new ArrayList<>();

                String model = Paths.get(inp).getFileName().toString();

                for (String word : words) {
                    output.add(new Word(model, word, w2v.topK(word, top)));
                }

                return output.stream();
            } catch (IOException e) {
                throw new IOError(e);
            }
        }).sorted()
          .collect(Collectors.groupingBy(w -> w.word));

        HashMap<String,Map<String,List<Word2vec.Result>>> word2model2list = new HashMap<>();
        modelWords.entrySet().forEach(e -> {

            HashMap<String,List<Word2vec.Result>> innerMap = new HashMap<>();
            e.getValue().forEach(w -> innerMap.put(w.model, w.ranks));

            word2model2list.put(e.getKey(), innerMap);
        });

        JtwigTemplate template = JtwigTemplate.classpathTemplate("template/report.twig");
        JtwigModel model = JtwigModel.newModel()
                                     .with("models", models.stream().map(inp -> Paths.get(inp).getFileName().toString()).collect(Collectors.toList()))
                                     .with("ranks", IntStream.range(0,top).boxed().collect(Collectors.toList()))
                                     .with("words", words)
                                     .with("data", word2model2list);
        try {
            FileOutputStream outf = new FileOutputStream(new File(args.get(1)));
            template.render(model, outf);
            outf.close();
        } catch (IOException e) {
            throw new IOError(e);
        }
    }
}
