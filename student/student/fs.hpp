#pragma once

#include <vector>
#include <string>

namespace fs {

/*!
 * \brief check whether the path exists or not
 * \param path  path
 * \return      true if exists else false
 */
bool IsExists(const std::string& path);

/*!
 * \brief make a directory
 * \note  if directory exists, do nothing and return true
 * \param dpath   directory path
 * \return        true if create success
 */
bool MakeDir(const std::string& dpath);

/*!
 * \brief fully remove a directory
 * \note  if directory not exists, do nothing and return true
 * \param dpath   directory path
 * \return        true if removed
 */
bool RemoveDir(const std::string& dpath);

/*!
 * \brief delete file
 * \note  if file doesn't exists, do nothing and return true
 * \param path  file path
 * \return      true if the file not exists
 */
bool DeleteFile(const std::string& path);

/*!
 * \brief list files in a directory with some file extensions
 * \param dpath   directory path
 * \param exts    extensions
 * \return        file list, None if dpath doesn't exists
 */
std::vector<std::string> ListDir(const std::string& dpath, const std::vector<std::string>& exts);

/*!
 * \brief list subdirectories in a given directory
 * \param dpath   directory path
 * \return        directories under the given directory path
 */
std::vector<std::string> ListSubDir(const std::string& dpath);

/*!
 * \brief move file
 * \param src_path  source file path
 * \param dst_path  dest file path
 * \return          true if move success
 */
bool MoveFile(const std::string& src_path, const std::string& dst_path);

/*!
 * \brief get file name, [/path/to/]xxx[.yyy] --> xxx
 * \param path  file path
 * \return      file name
 */
std::string GetFileName(const std::string& path);

/*!
 * \brief get file extension, [/path/to/]xxx.yyy --> yyy
 * \param path  file path
 * \return      file extension
 */
std::string GetFileExtension(const std::string& path);

}  // namespace fs
