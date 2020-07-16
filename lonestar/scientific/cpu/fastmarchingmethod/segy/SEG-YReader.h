#include "galois/Endian.h" // bswap32, bswap64
namespace galois {
static inline uint16_t bswap16(uint16_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap16(x);
#else
  return ((x << 8) & 0xff00) | ((x >> 8) & 0x00ff);
#endif
}

static inline uint16_t convert_be16toh(uint16_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return bswap16(x);
#else
  return x;
#endif
}
static inline uint32_t convert_be32toh(uint32_t x) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return bswap32(x);
#else
  return x;
#endif
}
} // end of namespace galois
///////////////////////////

struct BinaryFileHeader {
  uint32_t JOB_ID;   //!< (3201-3204) Job identification number.
  uint32_t LINE_NO;  //!< (3205-3208) Line number. For 3-D poststack data, this
                     //!< will typically contain the in-line number.
  uint32_t REEL_NUM; //!< (3209-3212) Reel number.
  uint16_t NUM_OF_TRACE;    //!< (3213-3214) Number of data traces per ensemble.
                            //!< Mandatory for prestack data.
  uint16_t NUM_OF_AUX;      //!< (3215-3216) Number of auxiliary traces per
                            //!< ensemble. Mandatory for prestack data.
  uint16_t INTERVAL_MS;     //!< (3217-3218) Sample interval. Microseconds (µs)
                            //!< for time data, Hertz (Hz) for frequency data,
                            //!< meters (m) or feet (ft) for depth data.
  uint16_t INTERVAL_MS_ORI; //!< (3219-3220) Sample interval of original field
                            //!< recording. Microseconds (µs) for time data,
                            //!< Hertz (Hz) for frequency data, meters (m) or
                            //!< feet (ft) for depth data.
  uint16_t NUM_OF_SAMPLES;  //!< (3221-3222) Number of samples per data trace.
                            //!< Note: The sample interval and number of samples
                            //!< in the Binary File Header should be for the
                            //!< primary set of seismic data traces in the file.
  uint16_t NUM_OF_SAMPLES_ORI;      //!< (3223-3224) Number of samples per data
                                    //!< trace for original field recording.
  uint16_t data_sample_format_code; //!< (3225-3226) Data sample format code.
                                    //!< Mandatory for all data.
  /*
   * 1 = 4-byte IBM floating-point
   * 2 = 4-byte, two's complement integer
   * 3 = 2-byte, two's complement integer
   * 4 = 4-byte fixed-point with gain (obsolete)
   * 5 = 4-byte IEEE floating-point
   * 6 = 8-byte IEEE floating-point
   * 7 = 3-byte twos complement integer
   * 8 = 1-byte, two's complement integer
   * 9 = 8-byte, two's complement integer
   * 10 = 4-byte, unsigned integer
   * 11 = 2-byte, unsigned integer
   * 12 = 8-byte, unsigned integer
   * 15 = 3-byte, unsigned integer
   * 16 = 1-byte, unsigned integer
   */
  uint16_t ENSEMBLE;  //!< (3227-3228) Ensemble fold  The expected number of
                      //!< data traces per trace ensemble (e.g. the CMP fold).
  int16_t TRACE_SORT; //!< (3229-3230) Trace sorting code (i.e. type of
                      //!< ensemble) :
                      /*
                       * -1 = Other (should be explained in a user Extended Textual File
                       * Header stanza)
                       * 0 = Unknown
                       * 1 = As recorded (no sorting)
                       * 2 = CDP ensemble
                       * 3 = Single fold continuous profile
                       * 4 = Horizontally stacked
                       * 5 = Common source point
                       * 6 = Common receiver point
                       * 7 = Common offset point
                       * 8 = Common mid-point
                       * 9 = Common conversion point
                       */
  uint16_t VERT_SUM;  //!< (3231-3232) Vertical sum code:
                      /*
                       * 1 = no sum,
                       * 2 = two sum,
                       * ...,
                       * N = M1 sum (M = 2 to 32,767)
                       */
  uint16_t SWEEP_FREQ_START;
  uint16_t SWEEP_FREQ_END;
  uint16_t SWEEP_LENGTH;
  uint16_t SWEEP_TYPE;
  uint16_t SWEEP_NUM_CHANNEL;
  uint16_t SWEEP_TAPER_LEN_START;
  uint16_t SWEEP_TAPER_LEN_END;
  uint16_t TAPER_TYPE;
  uint16_t CORRELATED;
  uint16_t BINARY_GAIN;
  uint16_t AMP_RECOR;
  uint16_t MEASURE_SYSTEM;
  uint16_t IMPULSE_POLAR;
  uint16_t POLAR_CODE;
  char UNNASSIGNED1[240];
  uint16_t SEGY_REV_NUM;
  uint16_t FIXED_LEN;
  uint16_t num_ext_tfh;
  char UNNASSIGNED2[94];

  void debugPrint() {
    galois::gDebug("Binary File Header:");
    galois::gDebug(JOB_ID, "\t (3201-3204) Job identification number.");
    galois::gDebug(LINE_NO,
                   "\t (3205-3208) Line number. For 3-D poststack data, "
                   "this will typically contain the in-line number.");
    galois::gDebug(REEL_NUM, "\t (3209-3212) Reel number.");
    galois::gDebug(NUM_OF_TRACE,
                   "\t (3213-3214) Number of data traces per ensemble. "
                   "Mandatory for prestack data.");
    galois::gDebug(NUM_OF_AUX, "\t (3215-3216) Number of auxiliary traces per "
                               "ensemble. Mandatory for prestack data.");
    galois::gDebug(INTERVAL_MS,
                   "\t (3217-3218) Sample interval. Microseconds (µs) "
                   "for time data, Hertz (Hz) for frequency data, "
                   "meters (m) or feet (ft) for depth data.");
    galois::gDebug(INTERVAL_MS_ORI,
                   "\t (3219-3220) Sample interval of original field "
                   "recording. Microseconds (µs) for time data, "
                   "Hertz (Hz) for frequency data, meters (m) or "
                   "feet (ft) for depth data.");
    galois::gDebug(NUM_OF_SAMPLES,
                   "\t (3221-3222) Number of samples per data trace."
                   "Note: The sample interval and number of samples "
                   "in the Binary File Header should be for the "
                   "primary set of seismic data traces in the file.");
    galois::gDebug(NUM_OF_SAMPLES_ORI,
                   "\t (3223-3224) Number of samples per data "
                   "trace for original field recording.");
    galois::gDebug(data_sample_format_code,
                   "\t (3225-3226) Data sample format code. Mandatory "
                   "for all data.");
    galois::gDebug(ENSEMBLE,
                   "\t (3227-3228) Ensemble fold  The expected number of"
                   "data traces per trace ensemble (e.g. the CMP fold).");
  }
};

struct TraceHeader {
  int TRACE_SEQ_GLOBAL;
  int TRACE_SEQ_LOCAL;
  int ORI_RECORD_NUM;
  int TRACE_NUM_FIELD;
  int SOURCE_POINT;
  int ENSEMBLE_NUM;
  int ENS_TRACE_NUM;
  short int TRACE_CODE;
  short int NUM_VERT_SUM;
  short int NUM_HORZ_SUM;
  short int DATA_USE;
  int DIST_CENT_RECV;
  int RECV_GRP_ELEV;
  int SURF_ELEV_SRC;
  int SOURCE_DEPTH;
  int DATUM_ELEV_RECV;
  int DATUM_ELAV_SRC;
  int WATER_DEPTH_SRC;
  int WATER_DEPTH_GRP;
  short int SCALE_DEPTH;
  short int SCALE_COOR;
  int SRC_COOR_X;
  int SRC_COOR_Y;
  int GRP_COOR_X;
  int GRP_COOR_Y;
  short int COOR_UNIT;
  short int WEATHER_VEL;
  short int SWEATHER_VEL;
  short int UPHOLE_T_SRC;
  short int UPHOLE_T_GRP;
  short int SRC_STA_CORRC;
  short int GRP_STA_CORRC;
  short int TOTAL_STA;
  short int LAG_TIME_A;
  short int LAG_TIME_B;
  short int DELAY_T;
  short int MUTE_T_STRT;
  short int MUTE_T_END;
  unsigned short int NUM_OF_SAMPL;
  unsigned short int SAMPLE_INTRVL;
  short int GAIN_TYPE;
  short int GAIN_CONST;
  short int GAIN_INIT;
  short int CORRLTD;
  short int SWEEP_FREQ_START;
  short int SWEEP_FREQ_END;
  short int SWEEP_LENGTH;
  short int SWEEP_TYPE;
  short int SWEEP_TAPER_LEN_START;
  short int SWEEP_TAPER_LEN_END;
  short int TAPER_TYPE;
  short int ALIAS_FREQ;
  short int ALIAS_SLOPE;
  short int NOTCH_FREQ;
  short int NOTCH_SLOPE;
  short int LOWCUT_FREQ;
  short int HIGHCUT_FREQ;
  short int LOWCUT_SLOPE;
  short int HIGHCUT_SLOPE;
  short int YEAR;
  short int DAY;
  short int HOUR;
  short int MINUTE;
  short int SECOND;
  short int TIME_CODE;
  short int WEIGHT_FACT;
  short int GEOPHNE_ROLL;
  short int GEOPHNE_TRACE;
  short int GEOPHNE_LAST;
  short int GAP_SIZE;
  short int OVER_TRAVEL;
  int ENS_COOR_X;
  int ENS_COOR_Y;
  int INLINE;
  int CROSS;
  int SHOOTPOINT;
  short int SHOOTPOINT_SCALE;
  short int TRACE_UNIT;
  char TRANSD_CONST[6];
  short int TRANSD_UNIT;
  short int TRACE_IDENT;
  short int SCALE_TIME;
  short int SRC_ORIENT;
  char SRC_DIRECTION[6];
  char SRC_MEASUREMT[6];
  short int SRC_UNIT;
  char UNNASSIGNED1[6];
};

#include "a2e.h"

#define HAS_TAPE_LABEL false

#define SEGY_TL_SIZE 128
#define SEGY_TFH_SIZE 3200
#define SEGY_TFH_NUM_COL 80
#define SEGY_TFH_NUM_ROW 40
#define SEGY_BFH_SIZE 400
#define SEGY_TH_SIZE 240

#include <filesystem>
///////////////////
// FileGraph.h
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
/**
 * Performs an mmap of all provided arguments.
 */
#ifdef HAVE_MMAP64
template <typename... Args>
void* mmap_big(Args... args) {
  return mmap64(std::forward<Args>(args)...);
}
//! offset type for mmap
typedef off64_t offset_t;
#else
template <typename... Args>
void* mmap_big(Args... args) {
  return mmap(std::forward<Args>(args)...);
}
//! offset type for mmap
typedef off_t offset_t;
#endif

/////////////////////

template <bool WITH_TAPE_LABEL>
class FileSEGY {
  static_assert(sizeof(char) == 1);
  std::unique_ptr<char[]> tape_label_128byte;
  std::unique_ptr<char[]> textual_file_header_3200byte;
  static_assert(sizeof(BinaryFileHeader) == SEGY_BFH_SIZE);
  std::unique_ptr<BinaryFileHeader> binary_file_header_400byte;
  std::unique_ptr<TraceHeader> trace_header_240byte;

public:
  FileSEGY()
      : tape_label_128byte(std::move(std::make_unique<char[]>(SEGY_TL_SIZE))),
        textual_file_header_3200byte(
            std::move(std::make_unique<char[]>(SEGY_TFH_SIZE))),
        binary_file_header_400byte(
            std::move(std::make_unique<BinaryFileHeader>())),
        trace_header_240byte(std::move(std::make_unique<TraceHeader>())) {}
  FileSEGY(const std::string& filename)
      : tape_label_128byte(std::move(std::make_unique<char[]>(SEGY_TL_SIZE))),
        textual_file_header_3200byte(
            std::move(std::make_unique<char[]>(SEGY_TFH_SIZE))),
        binary_file_header_400byte(
            std::move(std::make_unique<BinaryFileHeader>())),
        trace_header_240byte(std::move(std::make_unique<TraceHeader>())) {
    /*
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
      GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
    struct stat buf;
    if (fstat(fd, &buf) == -1)
      GALOIS_SYS_DIE("failed reading ", "'", filename, "'");

    // mmap file, then load from mem using fromMem function
    int _MAP_BASE = MAP_PRIVATE;
#ifdef MAP_POPULATE
    _MAP_BASE |= MAP_POPULATE;
#endif
    void* base = mmap_big(nullptr, buf.st_size, PROT_READ, _MAP_BASE, fd, 0);
    if (base == MAP_FAILED)
      GALOIS_SYS_DIE("failed reading ", "'", filename, "'");

    // fromMem(base, 0, 0, buf.st_size);
    textual_file_header_3200byte.reset((char*)base);
    binary_file_header_400byte.reset(
        (BinaryFileHeader*)((char*)base + SEGY_TFH_SIZE));
    */

    std::ifstream f;
    f.open(filename.c_str(), std::ios::in | std::ios::binary);
    if (f.fail()) {
      GALOIS_SYS_DIE("failed opening ", "'", filename, "'");
    }
    galois::gDebug("File size: ", std::filesystem::file_size(filename));

    // 1. Optional 128 byte SEG-Y Tape Label
    if (WITH_TAPE_LABEL) {
      f.read(reinterpret_cast<char*>(tape_label_128byte.get()),
             sizeof(char) * SEGY_TL_SIZE);
    }
    f.read(reinterpret_cast<char*>(textual_file_header_3200byte.get()),
           sizeof(char) * SEGY_TFH_SIZE);
    f.read(reinterpret_cast<char*>(binary_file_header_400byte.get()),
           sizeof(char) * SEGY_BFH_SIZE);
    if (true) {
      for (uint32_t i = 0; i < SEGY_TFH_SIZE; i++) {
        textual_file_header_3200byte[i] =
            EBCDICtoASCII(textual_file_header_3200byte[i]);
      }
    }

    binary_file_header_400byte->debugPrint();
    if (galois::convert_be16toh(binary_file_header_400byte->num_ext_tfh)) {
      /* read etfh */
      GALOIS_DIE("unimplemented");
    }

    f.read(reinterpret_cast<char*>(trace_header_240byte.get()),
           sizeof(char) * SEGY_TH_SIZE);

    f.close();
  }

  char* getTapeLabel() { return tape_label_128byte.get(); }
  char* getTextualFileHeader() { return textual_file_header_3200byte.get(); }
  BinaryFileHeader* getBinaryFileHeader() {
    return binary_file_header_400byte.get();
  }
};

#include <tuple>

template <bool WITH_TAPE_LABEL>
class SEGY {};

template <>
class SEGY<false> {
  std::array<std::string, SEGY_TFH_NUM_ROW> textual_file_header;
  uint16_t traceSortingCode;
  uint16_t sampleFormat;
  uint16_t numExtTFH;

public:
  SEGY() = delete;
  SEGY(FileSEGY<false>& raw) {
    char* tfh = raw.getTextualFileHeader();
    for (int i = 0, j = 0; i < SEGY_TFH_NUM_ROW && j < SEGY_TFH_SIZE;
         i++, j += SEGY_TFH_NUM_COL) {
      textual_file_header[i] = std::move(std::string(tfh, j, SEGY_TFH_NUM_COL));
      galois::gDebug(textual_file_header[i]);
    }

    BinaryFileHeader* bfh = raw.getBinaryFileHeader();

    traceSortingCode = galois::convert_be16toh(bfh->TRACE_SORT);
    galois::gDebug("TraceSortingCode: ", traceSortingCode);
    sampleFormat = galois::convert_be16toh(bfh->data_sample_format_code);
    galois::gDebug("SampleFormat: ", sampleFormat);
    numExtTFH = galois::convert_be16toh(bfh->num_ext_tfh);
    galois::gDebug("numExtTFH: ", numExtTFH);
  }
};

void ReadSegy(std::string filename) {
  // unsigned char* trace_header_240byte;
  FileSEGY<false> rawFile(filename);
  SEGY<false> segy(rawFile);

  //      galois::gDebug(tape_label_128byte
}
