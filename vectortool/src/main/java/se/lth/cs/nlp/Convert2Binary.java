package se.lth.cs.nlp;

import it.unimi.dsi.fastutil.floats.FloatArrayList;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.stream.IntStream;

/**
 * Created by csz-mkg on 2017-03-13.
 */
public class Convert2Binary {

    public static class Block {
        protected StringBuilder vocab = new StringBuilder();
        protected FloatArrayList vectordata = new FloatArrayList();

        protected byte[] rawdata;
        protected long start;
        protected long end;
        protected int region;
        protected int dim;

        public Block(long start, long end, int region, int dim, byte[] rawdata) {
            this.start = start;
            this.end = end;
            this.region = region;
            this.dim = dim;
            this.rawdata = rawdata;
        }
    }

    public static void process(Block block, boolean normalize) {
        String data = new String(block.rawdata, 0, block.rawdata.length, StandardCharsets.UTF_8);
        if(block.start == 0) {
            data = data.substring(data.indexOf('\n')+1);
        }

        if(normalize) {
            float[] buffs = new float[block.dim];
            for (String s : data.split("\n")) {
                String[] partial = s.split("\\s+");

                block.vocab.append(partial[0]).append('\n');

                double sum = 0.0f;
                for(int x = 1; x < partial.length; ++x) {
                    float v = Float.valueOf(partial[x]);
                    buffs[x-1] = v;
                    sum += v*v;
                }

                float v = (float)Math.sqrt(sum);
                if(v == 0.0) {
                    v = 1.0f;
                }

                for (int x = 0; x < buffs.length; x++) {
                    float tmp = buffs[x];
                    block.vectordata.add(tmp / v);
                }
            }
        }
        else {
            for (String s : data.split("\n")) {
                String[] partial = s.split("\\s+");

                block.vocab.append(partial[0]).append('\n');

                for(int x = 1; x < partial.length; ++x) {
                    block.vectordata.add(Float.valueOf(partial[x]));
                }
            }
        }
    }

    public static void process_old(String[] args) throws Exception {
        FileReader reader = new FileReader(args[0]);
        FileWriter vocab = new FileWriter(args[1] + ".vocab");
        FileOutputStream vecs = new FileOutputStream(args[1] + ".vecs");

        BufferedReader buf = new BufferedReader(reader);
        String line = buf.readLine();

        String[] header = line.split("\\s+");
        int vocabsize = Integer.parseInt(header[0]);
        int dim = Integer.parseInt(header[1]);

        vocab.write(String.valueOf(vocabsize));
        vocab.write("\n");

        ByteBuffer buff = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer buffer = buff.asIntBuffer();
        buffer.put(dim);
        vecs.write(buff.array());

        int k = 0;
        int i = 0;
        ByteBuffer vecBuffer = ByteBuffer.allocate(1024*dim*4).order(ByteOrder.LITTLE_ENDIAN);

        FloatBuffer floats = vecBuffer.asFloatBuffer();

        StringBuilder sb = new StringBuilder();
        System.out.println("Reading...");
        while((line = buf.readLine()) != null) {
            String[] partial = line.split("\\s+");

            sb.append(partial[0]).append('\n');

            float[] buffs = new float[dim];
            double sum = 0.0f;
            for(int x = 1; x < partial.length; ++x) {
                float v = Float.valueOf(partial[x]);
                buffs[x-1] = v;
                sum += v*v;
            }

            float v = (float)Math.sqrt(sum);
            if(v == 0.0) {
                v = 1.0f;
            }

            for (int x = 0; x < buffs.length; x++) {
                float tmp = buffs[x];
                floats.put(tmp / v);
            }

            i++;

            if(i == 1024) {
                System.out.print("\033[1A\033[2K"); // Erase line content
                k += 1024;
                System.out.println(String.format("Read %dK vectors.", k/1000));
                i = 0;
                floats.rewind();

                vecs.write(vecBuffer.array());
                vocab.write(sb.toString());

                sb.setLength(0);
            }
        }

        int pos = floats.position();
        floats.rewind();

        vocab.write(sb.toString());
        vecs.write(vecBuffer.array(),0,pos*4);

        vecs.close();
        vocab.close();
        System.out.println("Done.");
    }

    private static class Progress {
        public long numWords;
        public long numBytes;
        public int position;
        public long numTotalBytes;
        public long lastprintout;
    }

    public static void main(String[] args) throws Exception {
        convert(Arrays.asList(args));
    }

    public static void convert(List<String> args) throws Exception {

        String input = args.get(0);
        String output = args.get(1);
        boolean normalized = true;
        if(args.get(0).equals("-raw")) {
            normalized = false;
            input = args.get(1);
            output = args.get(2);
        }

        FileReader reader = new FileReader(input);
        BufferedReader buf = new BufferedReader(reader);
        String line = buf.readLine();
        String[] header = line.split("\\s+");

        int vocabsize = Integer.parseInt(header[0]);
        int dim = Integer.parseInt(header[1]);
        long size = new File(input).length();

        FileWriter vocab = new FileWriter(output + ".vocab");
        FileChannel binaryOutput = FileChannel.open(Paths.get(output + ".vecs"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);

        vocab.write(String.valueOf(vocabsize));
        vocab.write("\n");

        ByteBuffer buff = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        IntBuffer buffer = buff.asIntBuffer();
        buffer.put(dim);
        buff.position(4);
        buff.flip();
        binaryOutput.write(buff);

        buf.close();

        System.out.println("Scanning regions...");
        PartitionFile partitionFile = new PartitionFile(4*1024*1024, new File(input));
        LongArrayList regions = partitionFile.scan();
        System.out.println(String.format("Found %d regions.", regions.size()/2));

        RandomAccessFile raf = new RandomAccessFile(new File(input), "r");

        Object syncroot = new Object();

        Progress progress = new Progress();
        progress.numTotalBytes = size;
        progress.lastprintout = System.currentTimeMillis();

        final boolean usenormalization = normalized;

        System.out.println("Awaiting first progress update...");
        IntStream.range(0,regions.size()/2).parallel().forEach(reg -> {
            Block block;
            //Read data
            synchronized (syncroot) {
                long start = regions.getLong(progress.position*2);
                long end = regions.getLong(progress.position*2+1);
                progress.position += 1;

                byte[] rawbuffer = new byte[(int)(end-start)];
                try {
                    raf.seek(start);
                    raf.readFully(rawbuffer);
                } catch (IOException e) {
                    throw new IOError(e);
                }
                block = new Block(start,end,progress.position-1,dim,rawbuffer);
            }

            //Process
            process(block, usenormalization);
            ByteBuffer floatbuffer = ByteBuffer.allocate(block.vectordata.size()*4).order(ByteOrder.LITTLE_ENDIAN);
            floatbuffer.asFloatBuffer().put(block.vectordata.elements(), 0, block.vectordata.size());

            floatbuffer.position(floatbuffer.capacity());
            floatbuffer.flip();

            //Write data
            synchronized (syncroot) {
                try {
                    vocab.write(block.vocab.toString());
                    binaryOutput.write(floatbuffer);
                } catch (IOException e) {
                    throw new IOError(e);
                }

                long time = System.currentTimeMillis();
                progress.numBytes += block.end - block.start;
                progress.numWords += block.vectordata.size() / dim;

                if(time-progress.lastprintout > 250) {
                    progress.lastprintout = time;
                    System.out.print("\033[1A\033[2K"); // Erase line content
                    System.out.println(String.format("Read %dK vectors, %f%% completed.", progress.numWords / 1000, Math.round((progress.numBytes / Double.valueOf(progress.numTotalBytes)) * 10000.0) / 100.0));
                }
            }
        });

        System.out.print("\033[1A\033[2K"); // Erase line content
        System.out.println(String.format("Read %dK vectors, %f%% completed.", progress.numWords / 1000, Math.round((progress.numBytes / Double.valueOf(progress.numTotalBytes)) * 10000.0) / 100.0));

        binaryOutput.force(true);
        binaryOutput.close();

        vocab.close();

        System.out.println("Done.");
    }
}
