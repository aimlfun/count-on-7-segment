using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Count;

/// <summary>
/// Class to render a 7 segment display (257px x 340px)
/// </summary>
internal static class SevenSegmentDisplay
{
    /// <summary>
    /// Draws the chosen 7 segment display to an image.
    /// </summary>
    /// <param name="segments">Segments (in order) representing a-g.</param>
    internal static Bitmap Output(double[] segments)
    {
        if (segments == null || segments.Length != 7) throw new ArgumentException("7 segments are required for a 7 segment display.", nameof(segments));

        /* 7 segments are annotated as follows:
         * 
         *      [-a-]                 
         *   [|]     [|]
         *   [f]     [b]
         *   [|]     [|]
         *      [-g-]              
         *   [|]     [|]
         *   [e]     [c]            
         *   [|]     [|]   
         *      [-d-]     (#) 
         */

        Bitmap image = new(257, 340);

        using Graphics graphics = Graphics.FromImage(image);

        graphics.Clear(Color.Black);
        graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
        graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

        // add the (o) decimal point for completeness
        using SolidBrush brush = new(Color.FromArgb(15, 154, 139, 139));
        graphics.FillEllipse(brush, new Rectangle(200, 274, 37, 37));

        // light segments where segment contains a "1".
        for (int segment = 0; segment < segments.Length; segment++)
        {
            DrawSegmentOn7SegmentDisplay(segmentToLight: segment,
                                         illuminateSegment: Math.Round(segments[segment]) > 0.8F, 
                                         graphics); // 0.8 is a threshold to treat as 1
        }

        graphics.Flush();

        return image;
    }

    /// <summary>
    /// Draws a segment in the desired position either grey, or red.
    /// </summary>
    /// <param name="segmentToLight"></param>
    /// <param name="illuminateSegment"></param>
    /// <param name="graphics"></param>
    private static void DrawSegmentOn7SegmentDisplay(int segmentToLight, bool illuminateSegment, Graphics graphics)
    {
        // color RED if segment is to be illuminated, else greyish colour.
        using SolidBrush brushToPaintSegment = new(illuminateSegment ? Color.FromArgb(255, 255, 89, 99) : Color.FromArgb(15, 154, 139, 139));

        // the shapes for the segments I created by grabbing a photo from Google images, and working out where the points lie on it.
        Point[] pointsDefiningSegmentShape = Array.Empty<Point>(); // populated based on segment, then colour filled

        /* segments are       elements[]
         *     a                 0
         *   f    b            5   1
         *      g                6
         *   e    c            4   2
         *      d                3
         */
        switch (segmentToLight)
        {
            case 0: // a
                pointsDefiningSegmentShape = new Point[]
                {
                    new Point(85,43),
                    new Point(96,58),
                    new Point(203,58),
                    new Point(221,43),
                    new Point(217,36),
                    new Point(94,36)
                };
                break;

            case 1: // b
                pointsDefiningSegmentShape = new Point[]
                {
                    new Point(206,62),
                    new Point(188,153),
                    new Point(213,166),
                    new Point(232,55),
                    new Point(227,50)
                };
                break;

            case 2: // c
                pointsDefiningSegmentShape = new Point[]
                {
                    new Point(184,188),
                    new Point(168,273),
                    new Point(182,288),
                    new Point(192,280),
                    new Point(210,178)
                };
                break;

            case 3: // d
                pointsDefiningSegmentShape = new Point[]
                {
                    new Point(61,278),
                    new Point(163,278),
                    new Point(175,294),
                    new Point(166,303),
                    new Point(50,303),
                    new Point(42,294)
                };
                break;

            case 4: // e
                pointsDefiningSegmentShape = new Point[]
                {
                    new Point(56,274),
                    new Point(69,186),
                    new Point(48,175),
                    new Point(28,281),
                    new Point(37,289)
                };
                break;

            case 5: // f
                pointsDefiningSegmentShape = new Point[]
                {
                    new Point(79,52),
                    new Point(67,58),
                    new Point(49,165),
                    new Point(76,152),
                    new Point(92,66)
                };
                break;

            case 6: // g
                pointsDefiningSegmentShape = new Point[]
                {
                    new Point(59,169),
                    new Point(75,181),
                    new Point(182,181),
                    new Point(203,170),
                    new Point(185,158),
                    new Point(79,158)
                };
                break;
        }

        // paint the segment solid colour
        graphics.FillPolygon(brushToPaintSegment, pointsDefiningSegmentShape);

        // label the "segment" a-g, placed by computing center of the segment shape
        Point min = new(int.MaxValue, int.MaxValue);
        Point max = new(0, 0);

        // min and max of x/y directions enables the center to be determined (it's half way in both directions)
        foreach (Point point in pointsDefiningSegmentShape)
        {
            if (point.X < min.X) min.X = point.X;
            if (point.X > max.X) max.X = point.X;
            if (point.Y < min.Y) min.Y = point.Y;
            if (point.Y > max.Y) max.Y = point.Y;
        }

        using Font fontForLabel = new("Arial", 8);

        // 97 = "a".
        string label = $"{Char.ConvertFromUtf32(97 + segmentToLight)}";

        // centre is average of min+max. Think min=10, max=15, avg = (10+15)/2. We have to account for size of the label.
        SizeF sizeOfLabel = graphics.MeasureString(label, fontForLabel);

        graphics.DrawString(label, fontForLabel, Brushes.White, new PointF((min.X + max.X) / 2 - sizeOfLabel.Width / 2, (min.Y + max.Y) / 2 - sizeOfLabel.Height / 2));
    }
}