package se.lth.cs.nlp;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.longs.LongArrayList;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * Created by csz-mkg on 2017-03-14.
 */
public class PartitionFile {
    private int meanblocksize;
    private File input;
    private LongArrayList regions = new LongArrayList();

    public PartitionFile(int meanblocksize, File input) {
        this.meanblocksize = meanblocksize;
        this.input = input;
    }

    public LongArrayList scan() throws IOException {
        RandomAccessFile raf = new RandomAccessFile(input, "r");
        long totalLength = raf.length();
        long meanDistance = totalLength/meanblocksize;

        long start = 0;
        long end = meanDistance;
        byte[] buffer = new byte[65536];
        while(end < totalLength) {
            raf.seek(end);
            int read = raf.read(buffer);
            for(int i = 0; i < read; i++) {
                if(buffer[i] == '\n') {
                    //Found splitpoint
                    end = end + i;
                    regions.add(start);
                    regions.add(end);
                    start = end+1;
                    end = start+meanblocksize;
                    read = -2;
                    break;
                }
            }

            if(read == -1) {
                break;
            } else if(read >= 0) {
                end += read;
            }
        }

        if(start != totalLength) {
            regions.add(start);
            regions.add(totalLength);
        }

        return regions;
    }
}
