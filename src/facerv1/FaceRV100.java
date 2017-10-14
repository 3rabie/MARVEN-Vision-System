
package facerv1;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.face.*;
import org.opencv.utils.Converters;

public class FaceRV100  {

   



    static JFrame frame;
    static JLabel lbl;
    static ImageIcon icon;
    static JButton savedetectedface = new JButton("Save face");
    static JPanel panel = new JPanel(new BorderLayout(5, 5));
static Size size ;
    public static void main(String[] args) {
        System.load("/home/rabie/Downloads/opencv-master/build/lib/libopencv_java310.so");

        CascadeClassifier cascadeFaceClassifier = new CascadeClassifier(
                "/home/rabie/NetBeansProjects/FaceR-v1.0.0/src/data/haarcascades/haarcascade_frontalface_alt.xml");
        
        FaceRec face = new FaceRec();
        
        Size trainSize = face.loadTrainDir("/home/rabie/NetBeansProjects/FaceR-v1.0.0/src/data/persons/");
        System.out.println("facerec trained: " + (trainSize != null) + " !");
        
        VideoCapture Camera = new VideoCapture(0);

        if (Camera.isOpened()) {
            //to continue display the  without stop until close   
            int i = 1 ;
            while (true) {
                Mat frameCapture = new Mat();
                Camera.read(frameCapture);
                
                
                Mat frame_gray = new Mat();
                Imgproc.cvtColor(frameCapture, frame_gray, Imgproc.COLOR_BGRA2GRAY);
                Imgproc.equalizeHist(frame_gray, frame_gray);
                
                
                //load and convert the frames of video to detect faces 
                MatOfRect faces = new MatOfRect();
                cascadeFaceClassifier.detectMultiScale(frame_gray, faces);
                //draw a named rectungular surround detected faces 
                //FaceRecognizer fr = Face.createLBPHFaceRecognizer();
                //fr.train(null, frame_gray);
                Rect rect_Crop=null;
                for (Rect rect : faces.toArray()) {

                    Imgproc.rectangle(frameCapture, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                            new Scalar(0, 100, 0), 1);
                  //  Imgcodecs.imwrite(null, frame_gray);
                    rect_Crop = new Rect(rect.x, rect.y, rect.width, rect.height);
                    size = frame_gray.size();
                    Mat fi = frame_gray.submat(rect);
                    if (fi.size() != trainSize) // not needed for lbph, but for eigen and fisher
                    {
                        Imgproc.resize(fi, fi, trainSize);
                    }
                    if(trainSize != null){
                    String s = face.predict(fi);
                    if (s != "") {
                        Imgproc.putText(frameCapture, s, new Point(rect.x, rect.y - 5), 1, 2, new Scalar(0, 0, 255), 2);
                    }
                }
                if(savedetectedface.getModel().isPressed()){
                    Mat image_roi = new Mat(frameCapture,rect_Crop);
                    
                    Imgcodecs.imwrite("/home/rabie/NetBeansProjects/FaceR-v1.0.0/src/data/pic"+i+".png",image_roi);
                    i++;
                }
                PushImage(ConvertMat2Image(frameCapture));
                System.out.println(String.format("face detected ", faces.toArray().length));
            }
            } } else {
            System.out.println("Couldn't connect to Camera");
        }
                        Camera.release();

        savedetectedface.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

            }
        });
    }

    private static BufferedImage ConvertMat2Image(Mat VideoData) {

        MatOfByte byteMaT = new MatOfByte();
        Imgcodecs.imencode(".jpg", VideoData, byteMaT);
        byte[] byteArray = byteMaT.toArray();
        BufferedImage videoframe;
        try {
            InputStream in = new ByteArrayInputStream(byteArray);
            videoframe = ImageIO.read(in);
            
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        return videoframe;
    }

//create a frame to but the video on it 
    public static void FrameSetup() {
        frame = new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(700, 600);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    //method to check the frame and the image icon to display the video farmes on it 
    public static void PushImage(Image img2) {
        //check  the frame creation and display
        if (frame == null) {
            FrameSetup();
        }
        //check the lable to remove old frame(image) and put the new one 
        if (lbl != null) {
            frame.remove(lbl);
        }
        icon = new ImageIcon(img2);

        lbl = new JLabel();
        lbl.setIcon(icon);

        panel.add(savedetectedface, BorderLayout.CENTER);

        frame.add(lbl);
        frame.add(panel);

        //Frame validate update 
        frame.revalidate();
    }

}class FaceRec {

    FaceRecognizer fr = Face.createLBPHFaceRecognizer(2,8,8,8,90d);

    //
    // unlike the c++ demo, let's not mess with csv files, but use a folder on disk.
    //    each person should have its own subdir with images (all images the same size, ofc.)
    //   +- persons
    //     +- maria
    //       +- pic1.jpg
    //       +- pic2.jpg
    //     +- john
    //       +- pic1.jpg
    //       +- pic2.jpg
    //
    public Size loadTrainDir(String dir)
    {
        Size s = null;
        int label = 0;
        List<Mat> images = new ArrayList<Mat>();
        List<java.lang.Integer> labels = new ArrayList<java.lang.Integer>();
        File node = new File(dir);
        String[] subNode = node.list();
        for(String p : subNode){
            System.out.println(""+p);
        }
        if ( subNode==null ) return null;

        for(String person : subNode) {
          
            File subDir = new File(node, person);
            if ( ! subDir.isDirectory() ) continue;
            File[] pics = subDir.listFiles();
            for(File f : pics) {
                Mat m = Imgcodecs.imread(f.getAbsolutePath(),0);
                if (! m.empty()) {
                    images.add(m);
                    labels.add(label);
                    fr.setLabelInfo(label,subDir.getName());
                    s = m.size();
                }
            }
            label ++;
        }
        fr.train(images, Converters.vector_int_to_Mat(labels));
        return s;
    }
    public String predict(Mat img) {
        int[] id = {-1};
        double[] dist = {-1};
        fr.predict(img, id, dist);
        if (id[0] == -1) {
            return "Unknown";
        }else{
        double d = ((int) (dist[0] * 100));
        return fr.getLabelInfo(id[0]) ;
    }}
}