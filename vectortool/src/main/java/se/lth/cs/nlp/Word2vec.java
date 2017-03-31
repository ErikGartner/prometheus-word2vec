package se.lth.cs.nlp;

import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by marcusk on 2017-03-14.
 */
public class Word2vec {
    public static class Result {
        public final String word;
        public final float similarity;

        public Result(String word, float similarity) {
            this.word = word;
            this.similarity = similarity;
        }

        public String getWord() {
            return word;
        }

        public float getSimilarity() {
            return similarity;
        }
    }

    private static class Entry implements Comparable<Entry> {
        private int index;
        private float value;

        public Entry(int index, float value) {
            this.index = index;
            this.value = value;
        }

        @Override
        public int compareTo(Entry o) {
            return -Float.compare(value,o.value);
        }
    }

    private static INDArray load(int vocabsize, String path) throws IOException {
        long start = System.currentTimeMillis();
        System.out.println("Loading vectors...");
        FileChannel ch = FileChannel.open(Paths.get(path), StandardOpenOption.READ);
        ByteBuffer dim = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        ch.read(dim,0);
        dim.rewind();

        int dims = dim.asIntBuffer().get();
        FloatBuffer vecs = ch.map(FileChannel.MapMode.READ_ONLY, 4, ch.size()-4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[][] matrix = new float[vocabsize][dims];

        for(int k = 0; k < matrix.length; k++) {
            vecs.get(matrix[k]);
        }

        long end = System.currentTimeMillis();
        System.out.println("Done in " + (end-start) + " ms");
        System.out.println(String.format("Read %d vectors.", matrix.length));

        return Nd4j.create(matrix);
    }

    private final Object2IntOpenHashMap<String> wordIndex = new Object2IntOpenHashMap<>();
    private final List<String> words;
    private final int vocabsize;
    private final INDArray wordVectors;

    public Word2vec(File vocab, File vecs) throws IOException {
        long lstart = System.currentTimeMillis();
        List<String> lines = Files.lines(Paths.get(vocab.toURI())).collect(Collectors.toList());
        vocabsize = Integer.parseInt(lines.get(0));
        words = lines.subList(1,lines.size());

        System.out.println("Loading vocab " + vocab.getName() + "...");
        wordIndex.defaultReturnValue(-1);
        for (int i = 0; i < words.size(); i++) {
            wordIndex.put(words.get(i), i);
        }

        wordVectors = load(vocabsize, vecs.getAbsolutePath());
        long lend = System.currentTimeMillis();
        System.out.println("Fully loaded in " + (lend-lstart) + " ms");
    }

    public List<Result> topK(String word, int topn) {
        int idx = wordIndex.getInt(word);
        if(idx == -1)
            throw new IllegalArgumentException("word " + word + " could not be found!");

        INDArray vec = wordVectors.getRow(idx).transpose();
        INDArray weights = wordVectors.mmul(vec);
        Entry[] entries = new Entry[vocabsize];
        for (int i = 0; i < entries.length; i++) {
            entries[i] = new Entry(i, weights.getFloat(i));
        }

        Arrays.sort(entries);

        ArrayList<Result> result = new ArrayList<>(topn);
        for(int i = 0; i < topn; i++) {
            result.add(new Result(words.get(entries[i].index),entries[i].value));
        }

        return result;
    }
}
