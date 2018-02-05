//
// Created by xetql on 2/5/18.
//

#ifndef NBMPI_NBODY_IO_HPP
#define NBMPI_NBODY_IO_HPP

#include <vector>
#include <cstdio>
#include <cstdint>
#include <arpa/inet.h>
#include <vector>
/*@T
 * \section{Binary output}
 *
 * There are two output file options for our code: text and binary.
 * Originally, I had only text output; but I got impatient waiting
 * to view some of my longer runs, and I wanted something a bit more
 * compact, so added a binary option
 *
 * The viewer distinguishes the file type by looking at the first few
 * characters in the file: the tag [[NBView00]] means that what
 * follows is text data, while the tag [[NBView01]] means that what
 * follows is binary.  If you care about being able to read the
 * results of your data files across multiple versions of a code, it's
 * a good idea to have such a tag!
 *@c*/
#define VERSION_TAG "NBView01"

/*@T
 * Different platforms use different byte orders.  The Intel Pentium
 * hardware is little-endian, which means that it puts the least-significant
 * byte first -- almost like if we were to write a hundred twenty as 021 rather
 * than 120.  The Java system (which is where our viewer lives) is big-endian.
 * Big-endian ordering is also the so-called ``wire standard'' for sending
 * data over a network, so UNIX provides functions [[htonl]] and [[htons]]
 * to convert long (32-bit) and short (16-bit) numbers from the host
 * representation to the wire representation.  There is no corresponding
 * function [[htonf]] for floating point data, but we can construct
 * such a function by pretending floats look like 32-bit integers ---
 * the byte shuffling is the same.
 *@c*/
uint32_t htonf(void* data)
{
    return htonl(*(uint32_t*) data);
}

/*@T
 *
 * The header data consists of a count of the number of balls (a 32-bit
 * integer) and a scale parameter (a 32-bit floating poconst int number).
 * The scale parameter tells the viewer how big the view box is supposed
 * to be in the coordinate system of the simulation; right now, it is
 * always set to be 1 (i.e. the view box is $[0,1] \times [0,1]$)
 *@c*/
void write_header(FILE* fp, const int n)
{
    float scale = 1.0;
    uint32_t nn = htonl((uint32_t) n);
    uint32_t nscale = htonf(&scale);
    fprintf(fp, "%s\n", VERSION_TAG);
    fwrite(&nn,     sizeof(nn),     1, fp);
    fwrite(&nscale, sizeof(nscale), 1, fp);
}

/*@T
 *
 * The header data consists of a count of the number of balls (a 32-bit
 * integer) and a scale parameter (a 32-bit floating poconst int number).
 * The scale parameter tells the viewer how big the view box is supposed
 * to be in the coordinate system of the simulation; right now, it is
 * always set to be 1 (i.e. the view box is $[0,1] \times [0,1]$)
 *@c*/
void write_header(FILE* fp, const int n, float simsize)
{
    uint32_t nn = htonl((uint32_t) n);
    uint32_t nscale = htonf(&simsize);
    fprintf(fp, "%s\n", VERSION_TAG);
    fwrite(&nn,     sizeof(nn),     1, fp);
    fwrite(&nscale, sizeof(nscale), 1, fp);
}

void write_header(FILE* fp, const int n, double simsize)
{
    uint32_t nn = htonl((uint32_t) n);
    uint32_t nscale = htonf(&simsize);
    fprintf(fp, "%s\n", VERSION_TAG);
    fwrite(&nn,     sizeof(nn),     1, fp);
    fwrite(&nscale, sizeof(nscale), 1, fp);
}
/*@T
 *
 * After the header is a sequence of frames, each of which contains
 * $n_{\mathrm{particles}}$ pairs of 32-bit int floating point numbers.
 * There are no markers, end tags, etc; just the raw data.
 * The [[write_frame_data]] routine writes $n$ pairs of floats;
 * note that writing a single frame of output may involve multiple
 * calls to [[write_frame_data]]
 * Frame data just consists
 * integer) and a scale parameter (a 32-bit floating point number).
 * The scale parameter tells the viewer how big the view box is supposed
 * to be in the coordinate system of the simulation; right now, it is
 * always set to be 1 (i.e. the view box is $[0,1] \times [0,1]$)
 *@c*/
void write_frame_data(FILE* fp, const int n, float* x)
{
    for (int i = 0; i < n; ++i) {
        uint32_t xi = htonf(x++);
        uint32_t yi = htonf(x++);
        fwrite(&xi, sizeof(xi), 1, fp);
        fwrite(&yi, sizeof(yi), 1, fp);
    }
}

void write_frame_data(FILE* fp, const int n, std::vector<float> &x)
{
    for (int i = 0; i < n; ++i) {
        uint32_t xi = htonf(&x[i]);
        uint32_t yi = htonf(&x[i+1]);
        fwrite(&xi, sizeof(xi), 1, fp);
        fwrite(&yi, sizeof(yi), 1, fp);
    }
}void write_frame_data(FILE* fp, const int n, double* x)
{
    for (int i = 0; i < n; ++i) {
        uint32_t xi = htonf(x++);
        uint32_t yi = htonf(x++);
        fwrite(&xi, sizeof(xi), 1, fp);
        fwrite(&yi, sizeof(yi), 1, fp);
    }
}

void write_frame_data(FILE* fp, const int n, std::vector<double> &x)
{
    for (int i = 0; i < n; ++i) {
        uint32_t xi = htonf(&x[i]);
        uint32_t yi = htonf(&x[i+1]);
        fwrite(&xi, sizeof(xi), 1, fp);
        fwrite(&yi, sizeof(yi), 1, fp);
    }
}

#endif //NBMPI_NBODY_IO_HPP
