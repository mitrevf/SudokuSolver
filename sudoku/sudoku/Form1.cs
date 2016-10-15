using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV.OCR;
using Emgu.CV.UI;
using Emgu.Util;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;

namespace sudoku
{
    public partial class Form1 : Form
    {
        int c = 0;
        List<Image<Gray, Byte>> fields=new List<Image<Gray, byte>>();
        Tesseract tess;
        int[,] grid = new int[9, 9];
        List<Image<Gray, byte>> vzorci = new List<Image<Gray, byte>>();
        bool SolveSudoku(int[,] grid)
        {
            int row = 0;int col = 0;
            if (!FindUnassignedLocation(grid,ref row,ref col))
                return true; // success!
            for (int num = 1; num <= 9; num++)
            {
                if (isSafe(grid, row, col, num))
                {
                    grid[row,col] = num;
                    if (SolveSudoku(grid))
                        return true;
                    grid[row,col] = 0;
                }
            }
            return false;
        } 
        bool FindUnassignedLocation(int[,] grid, ref int row,ref int col)
        {
            for (row = 0; row < 9; row++)
                for (col = 0; col < 9; col++)
                    if (grid[row,col] == 0)
                        return true;
            return false;
        }
        bool UsedInRow(int[,] grid, int row, int num)
        {
            for (int col = 0; col < 9; col++)
                if (grid[row,col] == num)
                    return true;
            return false;
        }

        bool UsedInCol(int[,] grid, int col, int num)
        {
            for (int row = 0; row < 9; row++)
                if (grid[row,col] == num)
                    return true;
            return false;
        }
         bool UsedInBox(int[,] grid, int boxStartRow, int boxStartCol, int num)
        {
            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 3; col++)
                    if (grid[row + boxStartRow,col + boxStartCol] == num)
                        return true;
            return false;
        }

        bool isSafe(int[,] grid, int row, int col, int num)
        {
            return !UsedInRow(grid, row, num) &&
                   !UsedInCol(grid, col, num) &&
                   !UsedInBox(grid, row - row % 3, col - col % 3, num);
        }
        public Form1()
        {
            InitializeComponent();
            for (int i = 1; i < 10; i++)
                vzorci.Add(new Image<Gray, byte>(i.ToString() + "vzorec.png"));
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (SolveSudoku(grid) == false)
            {
                MessageBox.Show("neresljivo");
                return;
            }
            NarisiStevilke();
        }

        private void button2_Click(object sender, EventArgs e)  //procesiranje slike
        {
            c = 0;fields = new List<Image<Gray, byte>>(); grid = new int[9, 9];
            OpenFileDialog opf = new OpenFileDialog();
            if (opf.ShowDialog() != DialogResult.OK)
                return;
            Image<Gray, Byte> gray = new Image<Gray, byte>(opf.FileName);
            imageBox1.Image = gray.Clone();
            Image<Gray, Byte> izhod = new Image<Gray, byte>(gray.Width, gray.Height);
            // binariziranje(adaptivna pragovna segmentacija)
            izhod = gray.ThresholdAdaptive(new Gray(255), Emgu.CV.CvEnum.AdaptiveThresholdType.MeanC, Emgu.CV.CvEnum.ThresholdType.BinaryInv, 11, new Gray(11));
            izhod._SmoothGaussian(1); //glajenje, (ni obvezno potrebno)
           // imageBox2.Image = izhod.Clone();
            var nova = izhod.Clone();
            //posopek za iskanje najvecjega povezanega objekta
            VectorOfVectorOfPoint vvp = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(izhod, vvp, hierarchy, Emgu.CV.CvEnum.RetrType.Tree, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxNone);
            int largest_contour_index = 0;
            double largest_area = 0;
            VectorOfPoint largestContour;
            for (int i = 0; i < vvp.Size; i++)
            {
                double a = CvInvoke.ContourArea(vvp[i], false);  
                if (a > largest_area)
                {
                    largest_area = a;
                    largest_contour_index = i;               
                }
            }
            largestContour = vvp[largest_contour_index];
            Point[] lc = largestContour.ToArray();
            //iskanje kote za perspektivna transformacija
            Point topleft = new Point(gray.Width, gray.Height);
            Point topright = new Point(0, gray.Height);
            Point botright = new Point(0, 0);
            Point botleft = new Point(gray.Width, 0);
            foreach (Point p in lc)
            {
                if ((p.X + p.Y) < (topleft.X + topleft.Y))
                    topleft = p;
                else if ((p.X - p.Y) > (topright.X - topright.Y))
                    topright = p;
                else if ((p.X + p.Y) > (botright.X + botright.Y))
                    botright = p;
                else if ((p.Y - p.X) > (botleft.Y - botleft.X))
                    botleft = p;
            }
            //prerisemo gridlines, da se znebimo linije, potem ostanejo le stevilke(lazja razpoznava)
            CvInvoke.DrawContours(nova, vvp, largest_contour_index, new MCvScalar(0, 0, 0),6, Emgu.CV.CvEnum.LineType.EightConnected, hierarchy, 1);
            Image<Gray, Byte> warp = new Image<Gray, byte>(450, 450);
            PointF[] src = new PointF[] { topleft, topright, botright, botleft };
            PointF[] dst = new PointF[] { new Point(0, 0), new Point(450, 0), new Point(450, 450), new Point(0, 450) };
            Mat warpmat = CvInvoke.GetPerspectiveTransform(src, dst); //racunamo matrika za transformacija
            CvInvoke.WarpPerspective(nova, warp, warpmat, new Size(450, 450)); //izvedemo transformacija
            //imageBox1.Image = nova;
            imageBox2.Image = warp;
            //warp._Erode(1); //krcenje ali sirjenje, ni potrebno
            //warp._Dilate(1);

            //razpoznava stevilk, 2 moznosti (izbira so radiogumbov)
            if (radioButton1.Checked)
                tess = new Tesseract(@"C:/Emgu/emgucv-windows-universal 3.0.0.2157/bin/", null, OcrEngineMode.Default, "123456789 ");

            fields = new List<Image<Gray, byte>>(); //hranim polja za lazjo debagiranje
            for (int i = 0; i < 9; i++)
                for (int j = 0; j < 9; j++)
                {
                    Image<Gray, Byte> temp = (warp.GetSubRect(new Rectangle(j * 50+3, i * 50+3, 44, 44))).Clone(); //malo izpustimo po robu polja
                    temp._SmoothGaussian(1);
                    Gray sum = temp.GetSum(); //ce ni dovolj beli pikslov(dele objektov), ni stevilka.
                    if (sum.Intensity < 30000)
                        continue;

                    //spet iscemo najvecji element v polju, predvidevam da je stevilo
                    VectorOfVectorOfPoint vvptemp = new VectorOfVectorOfPoint();
                    Mat hierarchytemp = new Mat();
                    CvInvoke.FindContours(temp, vvptemp, hierarchytemp, Emgu.CV.CvEnum.RetrType.Tree, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxNone);
                    int ind = 0;
                    double area = 100;
                    VectorOfPoint contour;
                    for (int k = 0; k < vvptemp.Size; k++)
                    {
                        double ar = CvInvoke.ContourArea(vvptemp[k], false);  
                        if (ar > area)
                        {
                            area = ar;
                            ind = k;                
                        }
                    }
                    if (area == 100)
                        continue; //ce je najvecjega kontura area manj kot 100 (polje je 44x44), povzamemem da ni stevilo
                    contour = vvptemp[ind]; //kontura stevlike
                    
                    var tempimg = new Image<Gray, Byte>(44, 44, new Gray(0));
                    CvInvoke.DrawContours(tempimg, vvptemp, ind, new MCvScalar(255, 0, 0), -1, Emgu.CV.CvEnum.LineType.EightConnected, hierarchytemp);
                    //narisemo notranjosti najvecjega kontura v novi sliki z belo barvo
                    fields.Add(tempimg); //dodamo za pogled
                    if (radioButton2.Checked)
                    {
                        Rectangle br = CvInvoke.BoundingRectangle(contour);
                        int indeks = 0;
                        double vrednost = double.MaxValue;
                        for (int q = 0; q < 9; q++)
                        {   //racunamo podobnost s vsakem vzorcu
                            var kraj = tempimg.GetSubRect(new Rectangle(br.X, br.Y, vzorci[q].Width, vzorci[q].Height));
                            var pod = vzorci[q].AbsDiff(kraj);
                            var podobnost = pod.GetSum();
                            if (podobnost.Intensity < vrednost)
                            {
                                indeks = q + 1; //ker je zero based
                                vrednost = podobnost.Intensity;
                            }
                        }
                        grid[i, j] = indeks;//najbolj podobni je zaznana stevilka
                    }
                    else
                    {
                        tess.Recognize(tempimg); //raspoznava slike s pomocjo tesseract OCR vgrajen v openCV
                        var x = tess.GetCharacters();
                        if (x.Length == 1)
                            grid[i, j] = Convert.ToInt32(x[0].Text);
                    }
                }
            NarisiStevilke(); //izpisemo stevilki v polje
        }
        void NarisiStevilke() //metoda za risanje matrika s stevili v textbox
        {
            string tmp = "";
            for (int i=0;i<9;i++)
            {
                for (int j=0;j<9;j++)
                {
                    if (grid[i, j] != 0)
                        tmp += grid[i, j].ToString();
                    else
                        tmp += " ";
                    if (j != 8)
                        tmp += "\t";
                }
                if (i!=8)
                    tmp += "\n";
            }
            richTextBox1.Text = tmp;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (fields != null && fields.Count > 0)
            {
                imageBox1.Image = fields[c];
                c++;
                if (c == fields.Count)
                    c = 0;
            }
        }
    }
}
