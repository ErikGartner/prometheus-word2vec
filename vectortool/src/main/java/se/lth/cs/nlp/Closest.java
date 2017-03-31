package se.lth.cs.nlp;

import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.DataBuffer;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Primary Application
 */
public class Closest
{
    public static INDArray load(int vocabsize, String path) throws IOException {
        long start = System.currentTimeMillis();
        System.out.println("Loading vectors...");
        FileChannel ch = FileChannel.open(Paths.get(path + ".vecs"), StandardOpenOption.READ);
        long totalsz = ch.size();
        ByteBuffer dim = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        ch.read(dim,0);
        dim.rewind();

        int dims = dim.asIntBuffer().get();

        //Map 1M vectors at a time
        float[][] matrix = new float[vocabsize][dims];

        int k = 0;

        System.out.println("Start reading...");
        long sz = (vocabsize/(1024*1024))+1;
        for(long i = 0; i < sz; i++) {
            System.out.println("Read " + i + "M vectors.");
            long startpos = 4+i*1024*1024*dims*4;
            long endpos = Math.min(startpos+1024*1024*dims*4, totalsz);
            long partsz = endpos-startpos;

            FloatBuffer vecs = ch.map(FileChannel.MapMode.READ_ONLY, startpos, partsz).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            long left = partsz/dims/4;
            while(left > 0) {
                vecs.get(matrix[k++]);
                left--;
            }
        }

        long end = System.currentTimeMillis();
        System.out.println("Done in " + (end-start) + " ms");
        System.out.println(String.format("Read %d vectors.", matrix.length));

        return Nd4j.create(matrix);
    }

    public static class Entry implements Comparable<Entry> {
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

    public static void main(String[] args) throws Exception {
        closest(Arrays.asList(args));
    }

    public static void closest( List<String> args ) throws Exception
    {
        long lstart = System.currentTimeMillis();
        List<String> lines = Files.lines(Paths.get(args.get(0) + ".vocab")).collect(Collectors.toList());
        int vocabsize = Integer.parseInt(lines.get(0));
        lines = lines.subList(1,lines.size());

        System.out.println("Loading vocab...");
        Object2IntOpenHashMap<String> lookupIndex = new Object2IntOpenHashMap<>();
        lookupIndex.defaultReturnValue(-1);
        for (int i = 0; i < lines.size(); i++) {
            lookupIndex.put(lines.get(i),i);
        }

        INDArray mat = load(vocabsize, args.get(0));
        long lend = System.currentTimeMillis();
        System.out.println("Fully loaded in " + (lend-lstart) + " ms");

        BufferedReader bis = new BufferedReader(new InputStreamReader(System.in));
        String line;
        System.out.print("> ");
        System.out.flush();
        while((line = bis.readLine()) != null) {
            int idx = lookupIndex.getInt(line);
            if(idx == -1) {
                System.out.println("Could not find word: " + line);
            } else {
                long start = System.currentTimeMillis();
                INDArray vec = mat.getRow(idx).transpose();
                INDArray weights = mat.mmul(vec);
                Entry[] entries = new Entry[vocabsize];
                for (int i = 0; i < entries.length; i++) {
                    entries[i] = new Entry(i,weights.getFloat(i));
                }

                Arrays.sort(entries);
                for(int i = 0; i < 25; i++) {
                    System.out.println((i + 1) + ". " + lines.get(entries[i].index) + " = " + entries[i].value);
                }
                long end = System.currentTimeMillis();
                System.out.println("Computed in " + (end-start) + " ms.");
            }
            System.out.print("> ");
            System.out.flush();
        }
    }
}
