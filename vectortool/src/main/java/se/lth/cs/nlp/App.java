package se.lth.cs.nlp;

import java.util.Arrays;

/**
 * Created by csz-mkg on 2017-03-14.
 */
public class App {
    public static void main(String[] args) throws Exception {
        if(args.length == 0) {
            System.out.println("Usage: [convert|closest]");
            return;
        }

        if(args[0].equals("convert")) {
            Convert2Binary.convert(Arrays.asList(args).subList(1,args.length));
        }

        if(args[0].equals("closest")) {
            Closest.closest(Arrays.asList(args).subList(1,args.length));
        }
    }
}
